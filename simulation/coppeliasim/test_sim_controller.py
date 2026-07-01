"""
Unit tests for Milestone 4: sim_controller.py
Tests CoppeliaSim integration and command sending
"""

import numpy as np
import unittest
from unittest.mock import Mock, MagicMock, patch, call
import json
import zmq
from sim_controller import (
    CoppeliaSim,
    SimulationController,
    SimulationState,
    JointCommand,
    PositionCommand,
    get_simulator
)


class MockSocket:
    """Mock ZMQ socket for testing"""

    def __init__(self):
        self.messages_sent = []
        self.messages_to_receive = []
        self.timeout_error = False
        self.receive_index = 0

    def connect(self, address):
        pass

    def close(self):
        pass

    def setsockopt(self, option, value):
        pass

    def send_string(self, message):
        self.messages_sent.append(message)

    def recv_string(self):
        if self.timeout_error:
            raise zmq.error.Again("Timeout")

        if self.receive_index < len(self.messages_to_receive):
            msg = self.messages_to_receive[self.receive_index]
            self.receive_index += 1
            return msg
        return json.dumps({"status": "ok"})


class TestCoppeliaSim(unittest.TestCase):
    """Test suite for CoppeliaSim controller"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_socket = MockSocket()

    @patch('zmq.Context')
    def test_initialization(self, mock_context):
        """Test CoppeliaSim initialization"""
        sim = CoppeliaSim(address="localhost", port=23000, timeout=5.0)

        self.assertEqual(sim.address, "localhost")
        self.assertEqual(sim.port, 23000)
        self.assertEqual(sim.timeout, 5.0)
        self.assertEqual(sim.state, SimulationState.DISCONNECTED)

    @patch('zmq.Context')
    def test_state_transitions(self, mock_context):
        """Test simulation state transitions"""
        sim = CoppeliaSim()

        # Initial state
        self.assertEqual(sim.state, SimulationState.DISCONNECTED)
        self.assertFalse(sim.is_connected())
        self.assertFalse(sim.is_running())

        # Test manual state transitions
        sim.state = SimulationState.CONNECTED
        self.assertEqual(sim.state, SimulationState.CONNECTED)
        self.assertTrue(sim.is_connected())

        sim.state = SimulationState.RUNNING
        self.assertTrue(sim.is_running())

    @patch('zmq.Context')
    def test_start_simulation(self, mock_context):
        """Test starting simulation"""
        sim = CoppeliaSim(verbose=False)
        sim.socket = self.mock_socket
        sim.state = SimulationState.CONNECTED

        self.mock_socket.messages_to_receive = [
            json.dumps({"status": "ok"})
        ]

        success = sim.start_simulation()
        self.assertTrue(success)
        self.assertEqual(sim.state, SimulationState.RUNNING)
        self.assertTrue(sim.is_running())

    @patch('zmq.Context')
    def test_stop_simulation(self, mock_context):
        """Test stopping simulation"""
        sim = CoppeliaSim(verbose=False)
        sim.socket = self.mock_socket
        sim.state = SimulationState.RUNNING

        self.mock_socket.messages_to_receive = [
            json.dumps({"status": "ok"})
        ]

        success = sim.stop_simulation()
        self.assertTrue(success)
        self.assertEqual(sim.state, SimulationState.STOPPED)
        self.assertFalse(sim.is_running())

    @patch('zmq.Context')
    def test_pause_resume_simulation(self, mock_context):
        """Test pausing and resuming simulation"""
        sim = CoppeliaSim(verbose=False)
        sim.socket = self.mock_socket
        sim.state = SimulationState.RUNNING

        # Pause
        self.mock_socket.messages_to_receive = [
            json.dumps({"status": "ok"}),
            json.dumps({"status": "ok"})
        ]
        self.mock_socket.receive_index = 0

        success = sim.pause_simulation()
        self.assertTrue(success)
        self.assertEqual(sim.state, SimulationState.PAUSED)

        # Resume
        success = sim.resume_simulation()
        self.assertTrue(success)
        self.assertEqual(sim.state, SimulationState.RUNNING)

    @patch('zmq.Context')
    def test_set_joint_angle(self, mock_context):
        """Test setting joint angle"""
        sim = CoppeliaSim(verbose=False)
        sim.socket = self.mock_socket
        sim.state = SimulationState.CONNECTED

        self.mock_socket.messages_to_receive = [
            json.dumps({"status": "ok"})
        ]

        success = sim.set_joint_angle("joint1", 45.0, velocity=1.0)
        self.assertTrue(success)

        # Verify command was sent
        self.assertEqual(len(self.mock_socket.messages_sent), 1)
        command = json.loads(self.mock_socket.messages_sent[0])
        self.assertEqual(command["action"], "set_joint")
        self.assertEqual(command["joint"], "joint1")
        self.assertEqual(command["angle"], 45.0)

    @patch('zmq.Context')
    def test_set_object_position(self, mock_context):
        """Test setting object position"""
        sim = CoppeliaSim(verbose=False)
        sim.socket = self.mock_socket
        sim.state = SimulationState.CONNECTED

        self.mock_socket.messages_to_receive = [
            json.dumps({"status": "ok"})
        ]

        position = np.array([0.5, 0.3, -0.2])
        success = sim.set_object_position("gripper", position)
        self.assertTrue(success)

        # Verify command
        command = json.loads(self.mock_socket.messages_sent[0])
        self.assertEqual(command["action"], "set_object")
        self.assertEqual(command["object"], "gripper")
        np.testing.assert_array_almost_equal(command["position"], position)

    @patch('zmq.Context')
    def test_set_object_position_with_orientation(self, mock_context):
        """Test setting object position with orientation"""
        sim = CoppeliaSim(verbose=False)
        sim.socket = self.mock_socket
        sim.state = SimulationState.CONNECTED

        self.mock_socket.messages_to_receive = [
            json.dumps({"status": "ok"})
        ]

        position = np.array([0.5, 0.3, -0.2])
        orientation = np.array([0.0, 0.0, 45.0])
        success = sim.set_object_position("gripper", position, orientation)
        self.assertTrue(success)

        # Verify command includes orientation
        command = json.loads(self.mock_socket.messages_sent[0])
        self.assertIn("orientation", command)
        np.testing.assert_array_almost_equal(command["orientation"], orientation)

    @patch('zmq.Context')
    def test_get_joint_angle(self, mock_context):
        """Test getting joint angle"""
        sim = CoppeliaSim(verbose=False)
        sim.socket = self.mock_socket
        sim.state = SimulationState.CONNECTED

        self.mock_socket.messages_to_receive = [
            json.dumps({"status": "ok", "angle": 45.5})
        ]

        angle = sim.get_joint_angle("joint1")
        self.assertEqual(angle, 45.5)

    @patch('zmq.Context')
    def test_get_object_position(self, mock_context):
        """Test getting object position"""
        sim = CoppeliaSim(verbose=False)
        sim.socket = self.mock_socket
        sim.state = SimulationState.CONNECTED

        self.mock_socket.messages_to_receive = [
            json.dumps({"status": "ok", "position": [0.5, 0.3, -0.2]})
        ]

        position = sim.get_object_position("gripper")
        self.assertIsNotNone(position)
        np.testing.assert_array_almost_equal(position, np.array([0.5, 0.3, -0.2]))

    @patch('zmq.Context')
    def test_batch_joint_commands(self, mock_context):
        """Test sending batch joint commands"""
        sim = CoppeliaSim(verbose=False)
        sim.socket = self.mock_socket
        sim.state = SimulationState.CONNECTED

        self.mock_socket.messages_to_receive = [
            json.dumps({"status": "ok"}),
            json.dumps({"status": "ok"}),
            json.dumps({"status": "ok"})
        ]

        commands = [
            JointCommand("joint1", 45.0),
            JointCommand("joint2", -30.0),
            JointCommand("joint3", 60.0)
        ]

        success = sim.send_batch_joint_commands(commands)
        self.assertTrue(success)
        self.assertEqual(len(self.mock_socket.messages_sent), 3)

    @patch('zmq.Context')
    def test_reset_simulation(self, mock_context):
        """Test resetting simulation"""
        sim = CoppeliaSim(verbose=False)
        sim.socket = self.mock_socket
        sim.state = SimulationState.RUNNING

        self.mock_socket.messages_to_receive = [
            json.dumps({"status": "ok"})
        ]

        success = sim.reset_simulation()
        self.assertTrue(success)
        self.assertEqual(sim.state, SimulationState.STOPPED)

    @patch('zmq.Context')
    def test_execute_step(self, mock_context):
        """Test executing simulation step"""
        sim = CoppeliaSim(verbose=False)
        sim.socket = self.mock_socket
        sim.state = SimulationState.RUNNING

        self.mock_socket.messages_to_receive = [
            json.dumps({"status": "ok"})
        ]

        success = sim.execute_step(num_steps=5)
        self.assertTrue(success)

        # Verify command
        command = json.loads(self.mock_socket.messages_sent[0])
        self.assertEqual(command["steps"], 5)

    @patch('zmq.Context')
    def test_get_simulation_step(self, mock_context):
        """Test getting simulation step"""
        sim = CoppeliaSim(verbose=False)
        sim.socket = self.mock_socket
        sim.state = SimulationState.RUNNING

        self.mock_socket.messages_to_receive = [
            json.dumps({"status": "ok", "sim_step": 1000})
        ]

        step = sim.get_simulation_step()
        self.assertEqual(step, 1000)

    @patch('zmq.Context')
    def test_get_sensor_data(self, mock_context):
        """Test getting sensor data"""
        sim = CoppeliaSim(verbose=False)
        sim.socket = self.mock_socket
        sim.state = SimulationState.CONNECTED

        self.mock_socket.messages_to_receive = [
            json.dumps({"status": "ok", "data": {"force": 10.5, "distance": 0.3}})
        ]

        data = sim.get_sensor_data("force_sensor")
        self.assertIsNotNone(data)
        self.assertEqual(data["force"], 10.5)

    @patch('zmq.Context')
    def test_command_timeout(self, mock_context):
        """Test handling command timeout"""
        sim = CoppeliaSim(verbose=False)
        sim.socket = MagicMock()
        sim.state = SimulationState.CONNECTED

        sim.socket.recv_string.side_effect = zmq.error.Again("Timeout")

        result = sim.get_simulation_step()
        self.assertIsNone(result)

    @patch('zmq.Context')
    def test_disconnect(self, mock_context):
        """Test disconnecting from simulator"""
        sim = CoppeliaSim()
        sim.socket = self.mock_socket
        sim.context = MagicMock()
        sim.state = SimulationState.CONNECTED

        sim.disconnect()
        self.assertEqual(sim.state, SimulationState.DISCONNECTED)


class TestSimulationController(unittest.TestCase):
    """Test suite for SimulationController"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_sim = MagicMock(spec=CoppeliaSim)
        self.mock_mapper = MagicMock()

    def test_controller_initialization(self):
        """Test SimulationController initialization"""
        controller = SimulationController(self.mock_sim, self.mock_mapper)
        self.assertEqual(controller.sim, self.mock_sim)
        self.assertEqual(controller.mapper, self.mock_mapper)

    def test_apply_model_output_joint_angles(self):
        """Test applying model output to joint angles"""
        from mapper import OutputMapper, ControlSignal

        mapper = OutputMapper()
        controller = SimulationController(self.mock_sim, mapper)

        # Mock model output
        model_output = np.array([0.8, 0.1, 0.07, 0.03])

        success = controller.apply_model_output(model_output, output_type='joint_angles')

        # Should have called set_joint_angle for each joint
        self.assertTrue(self.mock_sim.set_joint_angle.called)
        call_count = self.mock_sim.set_joint_angle.call_count
        self.assertEqual(call_count, 3)  # 3 joints

    def test_apply_model_output_position(self):
        """Test applying model output to position"""
        from mapper import OutputMapper

        mapper = OutputMapper()
        controller = SimulationController(self.mock_sim, mapper)

        model_output = np.array([0.2, 0.3, 0.4, 0.1])

        success = controller.apply_model_output(model_output, output_type='position')

        # Should have called set_object_position
        self.assertTrue(self.mock_sim.set_object_position.called)

    def test_apply_model_output_without_mapper(self):
        """Test applying output without mapper"""
        self.mock_sim.verbose = False
        controller = SimulationController(self.mock_sim, mapper=None)
        model_output = np.array([0.8, 0.1, 0.07, 0.03])

        success = controller.apply_model_output(model_output)
        self.assertFalse(success)


class TestDataClasses(unittest.TestCase):
    """Test dataclass definitions"""

    def test_joint_command_creation(self):
        """Test JointCommand dataclass"""
        cmd = JointCommand("joint1", 45.0, target_velocity=0.5, max_force=50.0)
        self.assertEqual(cmd.joint_name, "joint1")
        self.assertEqual(cmd.angle, 45.0)
        self.assertEqual(cmd.target_velocity, 0.5)
        self.assertEqual(cmd.max_force, 50.0)

    def test_position_command_creation(self):
        """Test PositionCommand dataclass"""
        position = np.array([0.5, 0.3, -0.2])
        orientation = np.array([0.0, 0.0, 45.0])

        cmd = PositionCommand("gripper", position, orientation)
        self.assertEqual(cmd.object_name, "gripper")
        np.testing.assert_array_equal(cmd.position, position)
        np.testing.assert_array_equal(cmd.orientation, orientation)


class TestSimulationState(unittest.TestCase):
    """Test SimulationState enum"""

    def test_state_enum_values(self):
        """Test all state enum values exist"""
        states = [
            SimulationState.DISCONNECTED,
            SimulationState.CONNECTED,
            SimulationState.RUNNING,
            SimulationState.PAUSED,
            SimulationState.STOPPED
        ]
        self.assertEqual(len(states), 5)


class TestFactoryFunction(unittest.TestCase):
    """Test factory functions"""

    @patch('zmq.Context')
    def test_get_simulator(self, mock_context):
        """Test get_simulator factory function"""
        sim = get_simulator("localhost", 23000, verbose=False)
        self.assertIsInstance(sim, CoppeliaSim)
        self.assertEqual(sim.address, "localhost")
        self.assertEqual(sim.port, 23000)


if __name__ == '__main__':
    print("=" * 70)
    print("Running Milestone 4: CoppeliaSim Integration Tests")
    print("=" * 70)
    unittest.main(verbosity=2)
