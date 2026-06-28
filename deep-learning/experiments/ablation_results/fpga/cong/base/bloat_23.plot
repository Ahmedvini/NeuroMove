set title "bloat_23"
set palette negative rgbformula 21,22,23
set cbrange [0:2]
set yrange [360:0]
set size ratio 2.57143
set view map
splot "bloat_23.dat" using 1:2:3 with image
pause -1
