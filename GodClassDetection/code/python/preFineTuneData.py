# -*- encoding: utf-8 -*-
"""
@File    : preFineTuneData.py.py
@Time    : 2020-05-14 20:42
@Author  : knight
"""

# 随机数产生100个整数(0-100),放入一个列表中,统计出现次数最多的数字.
# 1.存放随机数列表

import random

# ant
# list = [27, 29, 30, 50, 118, 125, 178, 180, 183, 186, 205, 213, 308, 357, 358, 406, 418, 433, 438, 527, 661, 671, 717, 1081, 1082, 1083, 1084, 1085, 1086]
tmp = []
# ivy
# list = [95, 113, 218, 230, 346, 407, 484, 525, 580, 681]

# pig
# list = [2,
#         28,
#         666,
#         170,
#         468,
#         78]

# qpid
# list = [1421,
#         1207,
#         949,
#         1746,
#         334,
#         1406,
#         342,
#         974,
#         371,
#         157,
#         370,
#         1192,
#         377,
#         451,
#         1554,
#         1744,
#         1,
#         147,
#         1745,
#         130,
#         443,
#         ]

# list = [39, 77, 88, 134, 145, 271, 280, 286, 307, 320, 423, 451, 575, 656, 764, 788, 857, 997, 1006, 1031, 1092, 1162, 1214, 1332, 1378, 1470, 1525, 1553, 1582, 1617, 1667, 1739, 1743]

# struts
# list = [
#     864,
#     865,
#     188,
#     233,
#     661,
#     1265,
#     923,
#     628,
#     1619
# ]
# list = [33, 201, 405, 478, 519, 547, 602, 772, 805, 915, 1028, 1080, 1150, 1191, 1540, 1577]
# xerces
# list = [422,
#         368,
#         459,
#         289,
#         424,
#         457,
#         458,
#         179,
#         427,
#         236,
#         156,
#         460,
#         178,
#         187,
#         456,
#         133]

# list = [4, 26, 44, 63, 113, 135, 159, 179, 209, 233, 254, 273, 279, 295, 303, 342, 360, 369, 376, 385, 402, 423, 430, 443]

# freemind
# list = [
#     54,
#     281,
#     274,
#     394,
#     155,
#     343,
#     346,
#     34,
#     321,
#     135,
#     393
# ]

# list = [3, 25, 43, 82, 113, 137, 144, 161, 189, 212, 251, 273, 294, 310, 354, 385]

# jedit
# list = [
#     453,
#     504,
#     442,
#     508,
#     505,
#     506,
#     159,
#     58,
#     507,
#     169,
#     163,
#     461,
#     406,
#     181,
#     122]

# list = [11, 27,  57, 63, 73, 80, 122, 204, 220, 246, 270, 295, 362, 390, 398, 423, 441, 452, 477, 481, 496, 497, 501]

# jhotdraw
# list = [
#     549,
#     368,
#     431,
#     20,
#     18,
#     135,
#     502,
#     118,
#     275,
#     501,
#     302,
#     366
# ]
# list = [28, 43, 55, 70, 112, 124,  141, 158, 185, 200, 215, 287, 301, 341, 375, 406, 449, 499, 521]

list = []
list1 = []


def fun1():
    i = 15
    while i < 2380:
        list.append(i)
        i = i + 93
    # list.remove(0)
    # list.append(1)
    return list


def fun2():
    list1 = [27, 29, 30, 50, 118, 125, 178, 180, 183, 186, 205, 213, 308, 357,
             358, 406, 418, 433, 438, 527, 661, 671, 717, 1081, 1082, 1083, 1084, 1085, 1086]
    # list = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 176, 187, 198, 209, 220, 242, 253,
    #         264, 275, 286, 297, 319, 330, 341, 352, 363, 374, 385, 396, 407, 417, 429, 440, 451, 462, 473, 484, 495,
    #         506, 517, 528, 539, 550, 561, 572, 583, 594, 605, 616, 627, 638, 649, 660, 671, 682, 693, 704, 715, 726,
    #         737, 748, 759, 770, 792, 803, 814, 825, 836, 847, 858, 869, 880, 891, 902, 913, 924, 935, 946, 957,
    #         968, 979, 990]
    # list = [95, 113, 218, 230, 346, 407, 484, 525, 580, 681, 682]
    # list = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400,
    #         420, 440, 460, 480, 500, 520, 540, 560, 581, 600, 620, 640, 660]
    # list = [2, 28, 78, 170, 468, 666, 960, 961, 962]
    # list = [1, 130, 147, 157, 334, 342, 370, 371, 377, 443, 451, 949, 974, 1192, 1207, 1406, 1421, 1554, 1741, 1742, 1743, 1744, 1745, 1746]
    # list = [24, 48, 72, 96, 120, 144, 167, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504,
    #         528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984,
    #         1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392,
    #         1416, 1440, 1464, 1488, 1512, 1536, 1560, 1584, 1608, 1632, 1656, 1680, 1704, 1728]
    # list = [188, 233, 628, 661, 864, 865, 923, 1265, 1619, 1620]
    # list = [53, 106, 159, 212, 265, 318, 371, 424, 477, 530, 583, 636, 690, 742, 795, 848, 901, 954, 1007, 1060,
    #         1113, 1166, 1219, 1272, 1325, 1378, 1431, 1484, 1537, 1590]
    # list = [133, 156, 178, 179, 187, 236, 289, 368, 422, 424, 427, 456, 457, 458, 459, 460, 461]
    # list = [1, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99, 108, 117, 126, 135, 144, 153, 162, 171, 180, 189,
    #         198, 207, 216, 225, 234, 243, 252, 261, 270, 279, 288, 297, 306, 315, 324, 333, 342, 351,
    #         360, 369, 378, 387, 394, 405, 414, 423, 432, 441, 450]
    # list = [2384, 2385, 2386, 2387, 2388, 2389, 2390, 2391, 2392, 2393, 2394, 2395, 2396, 2397, 2398, 2399, 2400,
    #         2401, 2402, 2403, 2404, 2405, 2406, 2407, 2408, 2409]
    # list = [1, 31, 62, 93, 124, 155, 186, 217, 248, 279, 310, 341, 372, 403, 434, 465, 496, 527, 558, 589, 620, 651,
    #         682, 713, 744, 775, 806, 837, 868, 899, 930, 961, 992, 1023, 1054, 1085, 1116, 1147, 1178, 1209, 1240, 1271,
    #         1302, 1333, 1364, 1395, 1426, 1457, 1488, 1519, 1550, 1581, 1612, 1643, 1674, 1705, 1736, 1767, 1798, 1829,
    #         1860, 1891, 1922, 1953, 1984, 2015, 2046, 2077, 2108, 2139, 2170, 2201, 2232, 2263, 2294, 2325, 2356, 2380]
    # list = [34, 54, 135, 155, 274, 281, 321, 343, 346, 393, 394]
    # list = [1, 11, 22, 33, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 176, 187, 198, 209, 220, 231, 242, 253,
    #         264, 286, 297, 308, 319, 330, 352, 363, 374, 385]
    # list = [1, 39, 78, 117, 156, 195, 273, 312, 351, 390, 429, 468, 507, 546, 585, 624, 663, 701, 741, 780, 819, 858, 897, 936]
    # list = [122, 159, 163, 169, 181, 406, 442, 453, 461, 504, 505, 506, 507, 508, 509]
    # list = [18, 20, 118, 135, 275, 302, 366, 431, 501, 502, 548, 549]
    # list = [14, 28, 42, 56, 84, 98, 112, 126, 140, 168, 182, 196, 210, 224, 238, 252, 266, 280, 294,
    #         308, 322, 336, 350, 364, 378, 392, 406, 420, 434, 448, 462, 476, 490, 504, 518, 532]
    # list = [1805, 1806, 1807, 1808]
    # list = [1, 150, 300, 450, 600, 750, 900, 1050, 1200, 1350, 1500, 1650]
    # list = [37, 70, 116, 147, 192, 222, 268, 300, 335, 366, 400, 435, 467, 500, 532, 566, 600, 642, 675, 700, 730, 760,
    #         820, 850, 885, 918, 950, 1000, 1029]
    # list = [671, 672]
    # for i in range(le
    # list = [70, 130, 190, 250, 310, 370, 430, 490, 550, 610, 670]
    # list = [1, 120, 225, 330, 470, 580, 680, 785, 890]
    # list =[53, 125, 197, 269, 347, 413, 485, 557, 629, 701, 773, 845, 917, 989, 1061, 1133, 1205, 1277, 1349, 1421, 1493, 1565, 1637, 1690]
    # list = [70, 240, 400, 560, 730, 880, 1040, 1200, 1360, 1500]
    # list = [5, 32, 58, 85, 113, 148, 175, 202, 229, 256, 283, 310, 337, 364, 391, 418, 445]
    # list= [5, 38, 92, 126, 159, 192, 225, 258, 302, 335, 379]
    # list = [20, 137, 252, 369, 486, 603, 720, 837]
    # list = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
    #         200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440]
    # list = [5, 35, 65, 95, 125, 155, 185, 215, 245, 275, 305, 335, 365, 395, 425]
    # list = [21, 62, 120, 174, 216, 258, 300, 342, 384, 426, 468, 510]
    list = [100, 400, 800, 1400]

    # n(list1)):
    #     # print(list1[i] - 1)
    #     it = list1[i]
    #     list.append(it)
    return list


# list = fun1()
list = fun2()
list.sort()
print(len(list))
print(list)

file1 = open("/Users/knight/Desktop/GodClassDetection/trainset/test/Wicket/mn_train.txt")
file2 = open("/Users/knight/Desktop/GodClassDetection/trainset/test/Wicket/mt_train.txt")


def mn_train_1():
    num = 0
    index = 0
    # file1 = open("/Users/knight/Desktop/GodClassDetection-master-mao-new/trainset/fine_tune/Wicket/mn_train.txt")
    for line in file1:
        num = num + 1
        if num == list[index]:
            # print(num)
            index = index + 1
            # print(num)
            print(line)
            if index == len(list):
                print("文本信息over")
                break
        # num = num + 1
        # print(line)
    print(index)
    print(num)
    return num


def mn_train_2():
    num = 0
    index = 0
    # file2 = open("/Users/knight/Desktop/GodClassDetection-master-mao-new/trainset/fine_tune/Wicket/mt_train.txt")
    for line in file2:
        num = num + 1
        if num == list[index]:
            index = index + 1
            # print(num)
            print(line)
            if index == len(list):
                print("度量值over")
                break
            # index = index + 1
        # print(line)
    print(index)
    print(num)
    return num


def random_num(num):
    number = []
    # 2.循环100次
    n = len(list) * 2
    for i in range(0, n):
        m = random.randint(0, num)
        # 4.添加到列表中
        number.append(m)

    number.sort()
    print(number)
    print(len(number))

    for p in range(len(list)):
        for q in range(len(number)):
            # print(p)
            if number[q] == list[p]:
                number.remove(number[q])
            else:
                continue
            q = q + 1
        p = p + 1
    print("number列表为：")
    print(number)
    print(len(number))

    return number


def mn_train_0():
    # count = mn_train_1()
    # number = random_num(count)
    num = 0
    index = 0
    # file1 = open("/Users/knight/Desktop/GodClassDetection-master-mao-new/trainset/fine_tune/Apache Ant/mn_train.txt")
    for line in file1:
        num = num + 1
        if num == number[index]:
            tmp.append(number[index])
            index = index + 1
            print(line)
            if index == len(number):
                print("负样本文本信息")

    print(index)
    print(num)
    # print("负样本文本信息")


def mn_train_3():
    # count = mn_train_1()
    # number = random_num(count)
    num = 0
    index = 0

    # file2 = open("/Users/knight/Desktop/GodClassDetection-master-mao-new/trainset/fine_tune/Apache Ant/mt_train.txt")
    for line in file2:
        num = num + 1
        if num == number[index]:
            index = index + 1
            print(line)
            if index == len(number):
                print("负样本度量值over")

    print(index)
    print(num)
    # print("负样本度量值over")


if __name__ == '__main__':
    # count = mn_train_1()
    # number = random_num(count)
    # mn_train_1()
    # count = mn_train_1()
    # mn_train_2()
    # number = random_num(count)
    # mn_train_0()
    # mn_train_3()
    # print("EEEEEEEEEEEEEEE")
    # print(tmp)
    mn_train_1()
    mn_train_2()
