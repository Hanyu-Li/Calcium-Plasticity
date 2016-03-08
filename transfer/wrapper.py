from subprocess import call

sigma = 10

proc = call['./LIFstatic_arg', sigma]
output = proc.stdout.read()
print output
