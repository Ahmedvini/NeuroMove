-- ============================================================
-- BCI Exoskeleton Scene Builder — Lua Commander Script
-- ============================================================
-- HOW TO USE:
-- 1. In CoppeliaSim, make sure simulation is STOPPED (■)
-- 2. At the bottom of the CoppeliaSim window, find the
--    "Input code here" bar  (the Lua sandbox commander)
-- 3. Copy-paste THIS ENTIRE BLOCK into that bar and press Enter
-- 4. The arm will appear in the viewport immediately
-- 5. Then File → Save Scene As → scenes/bci_exo_scene.ttt
-- ============================================================

-- Geometry (metres)
local R           = 0.04
local UPPER_LEN   = 0.35
local FOREARM_LEN = 0.28
local HAND_LEN    = 0.12
local Z           = 0.8   -- height of arm above floor

-- Joint limits (radians)
local function deg(d) return d * math.pi / 180 end

-- ── Helper: remove object if it exists ───────────────────────
local function tryRemove(name)
    local ok, h = pcall(sim.getObject, '/' .. name)
    if ok and h >= 0 then
        sim.removeObjects({h}, false)
        print('  removed /' .. name)
    end
end

-- ── Helper: create a coloured cuboid ─────────────────────────
local function makeBox(name, size, pos, color)
    local h = sim.createPrimitiveShape(sim.primitiveshape_cuboid, size, 0)
    sim.setObjectAlias(h, name)
    sim.setObjectPosition(h, sim.handle_world, pos)
    sim.setShapeColor(h, nil, sim.colorcomponent_ambient_diffuse, color)
    return h
end

-- ── Helper: create revolute joint (position-control, dynamic) ─
local function makeJoint(name, pos, eulerXYZ, limLo, limHi)
    local h = sim.createJoint(
        sim.joint_revolute_subtype,
        sim.jointmode_dynamic,
        0
    )
    sim.setObjectAlias(h, name)
    sim.setObjectPosition(h, sim.handle_world, pos)
    if eulerXYZ then
        sim.setObjectOrientation(h, sim.handle_world, eulerXYZ)
    end
    -- joint range: [min, span]
    sim.setJointInterval(h, false, {limLo, limHi - limLo})
    -- enable position-control motor
    sim.setObjectInt32Param(h, sim.jointintparam_dynctrlmode,
                            sim.jointdynctrl_position)
    sim.setJointTargetPosition(h, 0)
    return h
end

-- ── Helper: attach child to parent keeping world pose ─────────
local function attach(child, parent)
    sim.setObjectParent(child, parent, true)
end

-- ============================================================
-- CLEAN UP previous build
-- ============================================================
print('── Removing old objects ─────────────────────────────────')
local names = {
    'exo_base',
    'exo_joint_shoulder', 'exo_upper_arm',
    'exo_joint_elbow',    'exo_forearm',
    'exo_joint_wrist',    'exo_hand',
    'exo_end_effector'
}
for _, n in ipairs(names) do tryRemove(n) end

-- ============================================================
-- BUILD
-- ============================================================
print('── Building scene ───────────────────────────────────────')

-- Joint world-X positions
local sx = 0.0
local ex = sx + UPPER_LEN
local wx = ex + FOREARM_LEN
local tx = wx + HAND_LEN

-- [1] Static base
local base = makeBox('exo_base',
    {0.12, 0.12, 0.12},
    {sx, 0, Z},
    {0.45, 0.45, 0.45})
sim.setObjectInt32Param(base, sim.shapeintparam_static, 1)
print('  [1] exo_base')

-- [2] Shoulder joint  (rotates around world-Z)
local jSh = makeJoint('exo_joint_shoulder',
    {sx, 0, Z},
    nil,          -- no orientation change; Z is default rotation axis
    deg(-90), deg(90))
attach(jSh, base)
print('  [2] exo_joint_shoulder')

-- [3] Upper arm
local ua = makeBox('exo_upper_arm',
    {UPPER_LEN, R*2, R*2},
    {sx + UPPER_LEN/2, 0, Z},
    {0.1, 0.35, 0.75})
attach(ua, jSh)
print('  [3] exo_upper_arm  (blue)')

-- [4] Elbow joint  (rotates around world-Z)
local jEl = makeJoint('exo_joint_elbow',
    {ex, 0, Z},
    nil,
    deg(0), deg(162))
attach(jEl, jSh)
print('  [4] exo_joint_elbow')

-- [5] Forearm
local fa = makeBox('exo_forearm',
    {FOREARM_LEN, R*1.7, R*1.7},
    {ex + FOREARM_LEN/2, 0, Z},
    {0.1, 0.65, 0.4})
attach(fa, jEl)
print('  [5] exo_forearm  (green)')

-- [6] Wrist joint  (rotates around world-X — roll)
--     rotate joint -90° around world-Y so its local-Z → world-X
local jWr = makeJoint('exo_joint_wrist',
    {wx, 0, Z},
    {0, -math.pi/2, 0},
    deg(-45), deg(45))
attach(jWr, jEl)
print('  [6] exo_joint_wrist')

-- [7] Hand
local hnd = makeBox('exo_hand',
    {HAND_LEN, 0.06, 0.04},
    {wx + HAND_LEN/2, 0, Z},
    {0.85, 0.45, 0.1})
attach(hnd, jWr)
print('  [7] exo_hand  (orange)')

-- [8] End-effector dummy
local ee = sim.createDummy(0.025)
sim.setObjectAlias(ee, 'exo_end_effector')
sim.setObjectPosition(ee, sim.handle_world, {tx, 0, Z})
attach(ee, jWr)
print('  [8] exo_end_effector  (dummy)')

-- ============================================================
-- VERIFY
-- ============================================================
print('── Verification ─────────────────────────────────────────')
local allOk = true
for _, name in ipairs(names) do
    local ok, h = pcall(sim.getObject, '/' .. name)
    if ok and h >= 0 then
        local p = sim.getObjectPosition(h, sim.handle_world)
        print(string.format('  ✓ /%-24s  x=%+.3f  z=%+.3f', name, p[1], p[3]))
    else
        print('  ✗ /' .. name .. '  NOT FOUND')
        allOk = false
    end
end

if allOk then
    print('\n  ✓ Scene built successfully!')
    print('  → Now do:  File → Save Scene As → scenes/bci_exo_scene.ttt')
else
    print('\n  [!] Some objects failed. Check errors above.')
end