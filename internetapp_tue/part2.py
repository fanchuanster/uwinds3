

my_items = ['gym mats', 'rigs', 'boxing gloves', 'ropes', 'treadmill', 'elliptical', 'dumbbell', 'yoga ball']

d1 = {}

with open("./catalog.txt") as fh:
    section_start = True
    a = []
    for line in fh:
        # remove the ending newline character
        line = line.strip()
        if not line:
            continue
        a.append(line)
        if len(a) == 3:
            d1[a[0]] = tuple(a[1:])
            a.clear()

for k in d1:
    if k not in my_items:
        print("'%s' is not available in the catalog" % k)
d1 = { k:v for k,v in d1.items() if k in my_items }
print(d1)

done = False

while not done:
    name = input("please enter a fitness item to order")
    if name in d1:
        n = 0
        while n <= 0:
            quantity = input("please enter how many items you would like to order")
            try:
                n = int(quantity)
            except Exception as e:
                print("only positive number is accepted", e)
                exit(1)
            if n <= 0:
                print('Negative values not accepted!')
            else:
                print("Your order is successful!")
        done = True
    else:
        print("'{}' is not a valid item name".format(name))