f = open("C:/Users/Jinsuby/Desktop/PycharmProjects/data/Kia/Kia_coco_thick_4/test_list.txt", "r")
read = f.read()
f.close()
f = open("C:/Users/Jinsuby/Desktop/PycharmProjects/data/Kia/Kia_coco_thick_4/trainval_list.txt", "w")
test_list = read.split()
print(test_list)
for i in range(1,104):
    if(str(i) not in test_list):
        f.write(str(i)+"\n")

f.close()