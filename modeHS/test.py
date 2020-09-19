from math import floor
mc, ext = 0,0
for i in range(50,200000):
	root = i**0.5
	a = floor(root)
	b = i - (root*root)
	mine = a + b/(3*a)
	ext = a + b/(4*a)
	if abs(mine-root)<abs(ext-root):
		mc += 1
		print(abs(mine-root))
	elif abs(ext-root)<abs(mine-root):
		ext += 1
		print(abs(ext-root))
	else:
		print(abs(mine-root), abs(ext-root))
print("Mine: ", mc, "\nExisting: ",ext)