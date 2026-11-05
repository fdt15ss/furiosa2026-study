from machine.car import drive
from machine.tv import watch

drive()
watch()

print("================================")
# from machine import car
# from machine import tv
from machine import car, tv


car.drive()
tv.watch()

print("============ test폴더 =============")
from machine.test import car
from machine.test import tv

car.drive()
tv.watch()