#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# v1.0.2

stock_factor = 3
const_budget_class1, const_budget_class2, const_budget_class3 = 500, 250, 100

def get_init_inventory():
    init_inventory = {
        1: {
            "price": 120,
            "margin": 8,
            "stock": 80*stock_factor,
            "category": "A"},
        2: {
            "price": 80,
            "margin": 7,
            "stock": 150*stock_factor,
            "category": "A"},
        3: {
            "price": 90,
            "margin": 12,
            "stock": 80*stock_factor,
            "category": "A"},
        4: {
            "price": 60,
            "margin": 10,
            "stock": 300*stock_factor,
            "category": "A"},
        5: {
            "price": 70,
            "margin": 11,
            "stock": 240*stock_factor,
            "category": "A"},
        6: {
            "price": 80,
            "margin": 9,
            "stock": 100*stock_factor,
            "category": "A"},
        7: {
            "price": 90,
            "margin": 10,
            "stock": 150*stock_factor,
            "category": "A"},
        8: {
            "price": 120,
            "margin": 8,
            "stock": 100*stock_factor,
            "category": "A"},
        9: {
            "price": 90,
            "margin": 6,
            "stock": 150*stock_factor,
            "category": "A"},
        10: {
            "price": 30,
            "margin": 12,
            "stock": 200*stock_factor,
            "category": "B"},
        11: {
            "price": 50,
            "margin": 11,
            "stock": 80*stock_factor,
            "category": "B"},
        12: {
            "price": 45,
            "margin": 15,
            "stock": 50*stock_factor,
            "category": "B"},
        13: {
            "price": 30,
            "margin": 15,
            "stock": 100*stock_factor,
            "category": "B"},
        14: {
            "price": 35,
            "margin": 9,
            "stock": 100*stock_factor,
            "category": "B"},
        15: {
            "price": 40,
            "margin": 13,
            "stock": 80*stock_factor,
            "category": "B"},
        16: {
            "price": 25,
            "margin": 10,
            "stock": 200*stock_factor,
            "category": "B"},
        17: {
            "price": 35,
            "margin": 14,
            "stock": 100*stock_factor,
            "category": "B"},
        18: {
            "price": 50,
            "margin": 6,
            "stock": 120*stock_factor,
            "category": "B"},
        19: {
            "price": 30,
            "margin": 10,
            "stock": 80*stock_factor,
            "category": "B"},
        20: {
            "price": 30,
            "margin": 11,
            "stock": 100*stock_factor,
            "category": "B"},
        21: {
            "price": 40,
            "margin": 8,
            "stock": 90*stock_factor,
            "category": "B"},
        22: {
            "price": 25,
            "margin": 7,
            "stock": 150*stock_factor,
            "category": "B"},
        23: {
            "price": 60,
            "margin": 10,
            "stock": 100*stock_factor,
            "category": "C"},
        24: {
            "price": 600,
            "margin": 9,
            "stock": 100*stock_factor,
            "category": "C"},
        25: {
            "price": 50,
            "margin": 11,
            "stock": 80*stock_factor,
            "category": "C"},
        26: {
            "price": 40,
            "margin": 10,
            "stock": 120*stock_factor,
            "category": "C"},
        27: {
            "price": 50,
            "margin": 14,
            "stock": 100*stock_factor,
            "category": "C"},
        28: {
            "price": 50,
            "margin": 13,
            "stock": 80*stock_factor,
            "category": "C"},
        29: {
            "price": 30,
            "margin": 8,
            "stock": 100*stock_factor,
            "category": "C"},
        30: {
            "price": 40,
            "margin": 9,
            "stock": 100*stock_factor,
            "category": "C"},
        31: {
            "price": 35,
            "margin": 12,
            "stock": 120*stock_factor,
            "category": "C"},
        32: {
            "price": 50,
            "margin": 20,
            "stock": 80*stock_factor,
            "category": "C"},
        33: {
            "price": 40,
            "margin": 10,
            "stock": 100*stock_factor,
            "category": "C"},
        34: {
            "price": 45,
            "margin": 14,
            "stock": 60*stock_factor,
            "category": "C"},
        35: {
            "price": 80,
            "margin": 12,
            "stock": 30*stock_factor,
            "category": "D"},
        36: {
            "price": 70,
            "margin": 8,
            "stock": 40*stock_factor,
            "category": "D"},
        37: {
            "price": 80,
            "margin": 8,
            "stock": 30*stock_factor,
            "category": "D"},
        38: {
            "price": 90,
            "margin": 6,
            "stock": 30*stock_factor,
            "category": "D"},
        39: {
            "price": 60,
            "margin": 15,
            "stock": 50*stock_factor,
            "category": "D"},
        40: {
            "price": 100,
            "margin": 6,
            "stock": 60*stock_factor,
            "category": "D"}
    }
    return init_inventory


def init_itemsets(cp, cf):
    inventory = cp.deepcopy(get_init_inventory())
        
    itemsets = {
        'item_1' : [
            [inventory[1]["price"], inventory[2]["price"], inventory[3]["price"], inventory[4]["price"], inventory[5]["price"], inventory[6]["price"],
              inventory[7]["price"], inventory[8]["price"], inventory[9]["price"]],
            [inventory[1]["margin"], inventory[2]["margin"], inventory[3]["margin"], inventory[4]["margin"], inventory[5]["margin"], inventory[6]["margin"],
              inventory[7]["margin"], inventory[8]["margin"], inventory[9]["margin"]],
            [0.4*cf, 0.15*cf, 0.1*cf, 0.3*cf, 0.3*cf, 0.4*cf, 0.3*cf, 0.3*cf, 0.3*cf],
            [400, 600, 400, 500, 150, 150, 150, 150, 150],
            [inventory[1]["stock"], inventory[2]["stock"], inventory[3]["stock"], inventory[4]["stock"], inventory[5]["stock"], inventory[6]["stock"],
              inventory[7]["stock"], inventory[8]["stock"], inventory[9]["stock"]],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
        ],
        
        'item_2' : [
            [inventory[11]["price"], inventory[15]["price"], inventory[13]["price"]],
            [inventory[11]["margin"], inventory[15]["margin"], inventory[13]["margin"]],
            [0.15*cf, 0.05*cf, 0.4*cf],
            [50, 70, 60],
            [inventory[11]["stock"], inventory[15]["stock"], inventory[13]["stock"]],
            [1, 2, 3],
            [11, 15, 13]
        ],
        
        'item_3' : [
            [inventory[37]["price"]],
            [inventory[37]["margin"]],
            [0.2*cf],
            [50],
            [inventory[37]["stock"]],
            [1],
            [14]
        ],
        
        'item_4' : [
            [inventory[5]["price"], inventory[7]["price"], inventory[8]["price"]],
            [inventory[5]["margin"], inventory[7]["margin"], inventory[8]["margin"]],
            [0.30*cf, 0.1*cf, 0.2*cf],
            [20, 20, 30],
            [inventory[5]["stock"], inventory[7]["stock"], inventory[8]["stock"]],
            [1, 2, 3],
            [5, 7, 8]
        ],
        
        'item_5' : [
            [inventory[21]["price"], inventory[12]["price"], inventory[15]["price"], inventory[14]["price"]],
            [inventory[21]["margin"], inventory[4]["margin"], inventory[15]["margin"], inventory[14]["margin"]],
            [0.2*cf, 0.3*cf, 0.3*cf, 0.25*cf],
            [20, 20, 30, 40],
            [inventory[21]["stock"], inventory[12]["stock"], inventory[15]["stock"], inventory[14]["stock"]],
            [1, 2, 3, 4],
            [21, 12, 15, 14]
        ],
        
        'item_6' : [
            [inventory[1]["price"], inventory[2]["price"], inventory[4]["price"]],
            [inventory[1]["margin"], inventory[2]["margin"], inventory[4]["margin"]],
            [0.2*cf, 0.4*cf, 0.15*cf],
            [50, 70, 60],
            [inventory[1]["stock"], inventory[2]["stock"], inventory[4]["stock"]],
            [1, 2, 3],
            [1, 2, 4]
        ],
        
        'item_7' : [
            [inventory[35]["price"], inventory[34]["price"], inventory[38]["price"], inventory[37]["price"]],
            [inventory[35]["margin"], inventory[34]["margin"], inventory[38]["margin"], inventory[37]["margin"]],
            [0.2*cf, 0.3*cf, 0.3*cf, 0.25*cf],
            [20, 20, 30, 40],
            [inventory[35]["stock"], inventory[34]["stock"], inventory[38]["stock"], inventory[37]["stock"]],
            [1, 2, 3, 4],
            [35, 34, 38, 37]
        ],
        
        'item_8' : [
            [inventory[27]["price"], inventory[29]["price"]],
            [inventory[27]["margin"], inventory[29]["margin"]],
            [0.4*cf, 0.6*cf],
            [50, 50],
            [inventory[27]["stock"], inventory[29]["stock"]],
            [1, 2],
            [27, 8]
        ],
        
        'item_9' : [
            [inventory[39]["price"], inventory[36]["price"], inventory[35]["price"], inventory[40]["price"], inventory[37]["price"], inventory[38]["price"]],
            [inventory[39]["margin"], inventory[36]["margin"], inventory[35]["margin"], inventory[40]["margin"], inventory[37]["margin"], inventory[38]["margin"]],
            [0.4*cf, 0.15*cf, 0.1*cf, 0.3*cf, 0.3*cf, 0.4*cf],
            [400, 600, 400, 500, 150, 150],
            [inventory[39]["stock"], inventory[36]["stock"], inventory[35]["stock"], inventory[40]["stock"],inventory[37]["stock"], inventory[38]["stock"]],
            [1, 2, 3, 4, 5, 6],
            [39, 36, 35, 40, 37, 38]
        ],
        
        'item_10' : [
        [inventory[17]["price"], inventory[20]["price"]],
        [inventory[17]["margin"], inventory[20]["margin"]],
        [0.1*cf, 0.2*cf],
        [50, 50],
        [inventory[17]["stock"], inventory[20]["stock"]],
        [1, 2],
        [17, 20]
        ]
    }

    
    return itemsets

def get_customers(np, cp, p_split_c1_c23, p_split_c2_c3, g_factor, const_max_cust_num, cfinit = 1):
    customers = []
    const_min_cust_num = int(round(const_max_cust_num * 0.66, 0))
    const_max_cust_num, const_min_cust_num = int(round(const_max_cust_num * g_factor, 0)), int(round(const_min_cust_num * g_factor, 0))
    num_cust = -1
    while (num_cust > const_max_cust_num) or (num_cust < const_min_cust_num):
        num_cust = int(np.random.normal((sum([const_min_cust_num, const_max_cust_num])/len([const_min_cust_num, const_max_cust_num])), int(const_max_cust_num/const_min_cust_num)))
    for i in range(0, num_cust):
        budget, cclass = -1, -1
        if  np.random.binomial(1, p_split_c1_c23) == 1:
            budget, cclass = np.random.normal(const_budget_class1), 1
        else:
            if np.random.binomial(1, p_split_c2_c3) == 1:
                budget, cclass = np.random.normal(const_budget_class2), 2
            else:
                budget, cclass = np.random.normal(const_budget_class3), 3
        num_item_cats = -1
        while (num_item_cats > 9) or (num_item_cats < 1):
            num_item_cats = int(np.random.normal(6, 2))
        item_index = []
        for k in range(num_item_cats):
            k_item = -1
            cf = cfinit
            if cclass == 1:
                cf = cf*0.75
            if cclass == 3:
                cf = cf*1.25
            item_sets = init_itemsets(cp, cf)
            while (k_item > (len(item_sets)-1)) or (k_item < 0) or (k_item in item_index):
                k_item = np.random.randint((len(item_sets)-1))
            item_index.append(k_item)
        itemset = [item_sets['item_'+str(k+1)] for k in item_index]
        customers.append([itemset, int(budget), cclass])
    return customers


