#!/usr/bin/env python
# coding: utf-8
# v1.0.6

import gurobipy as gp; from gurobipy import GRB
import numpy as np, scipy.stats as st, matplotlib.pyplot as plt, copy, time, datetime
import Master_Thesis_demand_simulator as ds


# In[]:


print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
starttime = datetime.datetime.fromtimestamp(time.time())    

# SIMULATION SETTINGS
t_periods = 1000
sim_runs = 90

# PARAMTERS, VARIABLES
class_penalty_list = [[0,0,0],[10,0,0],[10,2,0]]
param_cost_return_list = [1]
param_avg_days_to_return_list = [10]
param_restock_policy_list = [20]
param_stock_class_threshold_list = [20]

param_demand_class_threshold = 1
param_penalty_after_n_ranks = 1

# CONSTANTS
const_value_per_box = 500
const_max_num_items = 10
const_max_cust_num = 30
const_penalty_adjustment = 1e-3



grid_size = (len(class_penalty_list)*sim_runs*len(param_cost_return_list)*len(param_restock_policy_list)*len(param_avg_days_to_return_list)*len(param_stock_class_threshold_list))
grid_cnt = 0


print_keys = ['0_0_0', '10_0_0', '10_2_0']

for param_stock_class_threshold in param_stock_class_threshold_list:
    for param_restock_policy in param_restock_policy_list:
        for param_avg_days_to_return in param_avg_days_to_return_list:
            for param_cost_return in param_cost_return_list:        
                
                result = {}; sim_profit_results, sim_profit_interval, confidence_interval = {}, {}, {}
                
                for debug1 in range(0,1): # debug
                    
                    for debug2 in range(0,1): # debug
                        
                        for class_penalty in class_penalty_list:
                            
                            penalty1 = class_penalty[0]
                            penalty2 = class_penalty[1]
                            penalty3 = class_penalty[2]
                
                            penalty_combination = str(penalty1)+str('_')+str(penalty2)+str('_')+str(penalty3)
                            run_profits = []
                            
                            for run in range(0, sim_runs):
                                grid_cnt += 1
                                
                                g_factor = 1; g_addition = 0.001
                                
                                p_split_c1_c23, p_split_c2_c3 = 0.2, 0.5
                                t_p_customers_c1, t_p_customers_c2 = [], []
                                t_profits = {}
                                t_items = {}
                                
                                # In[]:
                                
                                inventory = copy.deepcopy(ds.get_init_inventory())
                                
                                
                                # In[]:
                                
                                
                                # i = customer index
                                # x = customer data category index --> items = 0
                                # k = item category index
                                # l = item data category index
                                # n = item index
                                
                                # customers[i][x][k][l][n]
                                
                                # item[l=0] -> prices
                                # item[l=1] -> margins
                                # item[l=2] -> p_returns
                                # item[l=3] -> volumes
                                # item[l=4] -> inventory
                                # item[l=5] -> rank
                                # item[l=6] -> ID
                                
                                # customers[i][0] #items
                                # customers[i][1] #budget
                                # customers[i][2] #customer class
                                
                                
                                for t in range(1, t_periods+1):
                                    
                                    customers = ds.get_customers(np, copy, p_split_c1_c23, p_split_c2_c3, g_factor, const_max_cust_num)
                                    if t % param_restock_policy == 0:
                                        inventory = copy.deepcopy(ds.get_init_inventory())

                                
                                    # In[]:
                                
                                    try:
                                        model = False
                                        del model
                                    except:
                                        pass
                                    model = gp.Model()

                                
                                    # In[]:
                                
                                    # decision variable on item level
                                    x_items = []
                                    y_penalty = []
                                    for i, i_item_list in enumerate(customers):
                                        inner_x_items = []
                                        y_penalty.append(model.addVars(range(0, len(i_item_list[0])), ub=1, lb=0,
                                                                       name=(str(i)+'_penalty_var'), vtype=GRB.BINARY))
                                        for k, k_item_list in enumerate(customers[i][0]):
                                            inner_x_items.append(model.addVars(range(0, len(k_item_list[0])),
                                                                               name=(str(i)+'_'+str(k)+'_items_decision'), vtype=GRB.BINARY))
                                        x_items.append(inner_x_items)
                                
                                
                                    # In[]:
                                
                                    # low-inventory items class constraint
                                    # if inventory level for item < threshold for customer class --> inventory decision set to 0
                                    # helper list for summing 0s in x_items[i][k][n] == 0 per customer
                                    constraint_inventory_level = []
                                    for i in range(len(x_items)):
                                        constraint_inventory_level.append(0)
                                        if customers[i][2] > param_demand_class_threshold:
                                            for k in range(0, (len(x_items[i]))):
                                                for n in range(0, (len(x_items[i][k]))):
                                                    if customers[i][0][k][4][n] <= param_stock_class_threshold:
                                                        model.addConstr(x_items[i][k][n] == 0,
                                                                        name=(str(i)+str(k)+"_stock_level_class"))
                                                        constraint_inventory_level[i] += 1
                                
                                    # In[]:
                                
                                    inventory_limit_dict = {}
                                    for i, xi_items in enumerate(x_items):
                                        for k in range(0, (len(x_items[i]))):
                                            for n in range(0, (len(x_items[i][k]))):
                                
                                                if customers[i][0][k][6][n] in inventory_limit_dict.keys():
                                                    inventory_limit_dict[customers[i][0]
                                                                         [k][6][n]] += x_items[i][k][n]
                                                else:
                                                    inventory_limit_dict[customers[i][0][k][6][n]] = 0
                                                    inventory_limit_dict[customers[i][0]
                                                                         [k][6][n]] += x_items[i][k][n]
                                
                                    # In[]:
                                
                                    # max shipped items in one period = max items available
                                    constr_stocklevel = model.addConstrs((gp.quicksum([inventory_limit_dict[key]]) <=
                                                                          inventory[key]['stock'] for key in inventory_limit_dict.keys()),
                                                                         name=("item_stocklevels"))
                                
                                    # In[]:
                                
                                    # max budget constraint
                                    for i in range(len(x_items)):
                                        i_budget = customers[i][1]
                                        for k, k_items in enumerate(x_items[i]):
                                
                                            model.addConstr(gp.quicksum(x_items[i][k][n]*(1-customers[i][0][k][2][n]*customers[i][0][k][0][n]) for n in range(0, len(x_items[i][k]))) <= i_budget,
                                                            name=(str(i)+str(k)+"_budget_per_customer"))  # budget for each customer over all items
                                
                                
                                    # I[8]:
                                    # Penalty 
                                    for i, xi_items in enumerate(x_items):
                                        for k, xk_items in enumerate(x_items[i]):
                                            
                                            model.addConstrs((y_penalty[i][k] == (1-xk_items[n]) for n in range(0, min(param_penalty_after_n_ranks, len(xk_items)))),
                                                             name=(str(i)+"_"+str(k)+"_penalty_x_constraint"))
                            
                                            
                                    # In[]:
                                
                                    # maximum number of items per box / assumption: item categories per customer <= const_max_num_items
                                    # for each customer sum of x <= 10
                                    for i, xi_items in enumerate(x_items):
                                
                                        i_temp_item_list = []
                                        [[i_temp_item_list.append(xi_items[k][n]) for n in range(
                                            0, len(xi_items[k]))] for k, xk_items in enumerate(xi_items)]
                                        model.addConstr(gp.quicksum(i_temp_item_list) <= const_max_num_items,
                                                        name=(str(i)+"_max_number_of_items_per_box"))
                                

                                    # In[]:
                                
                                    # budget per customer constraint / expected spending
                                    # for each customer price * probability * decision
                                    for i in range(len(x_items)):
                                        i_budget = customers[i][1]
                                        for k, k_items in enumerate(x_items[i]):
                                            model.addConstr(gp.quicksum(x_items[i][k][n]*(1-customers[i][0][k][2][n]*customers[i][0][k][0][n]) for n in range(0, len(x_items[i][k]))) <= i_budget,
                                                            name=(str(i)+str(k)+"_budget_per_customer"))  # budget for each customer over all items
                                
                                    # In[]:
                                
                                    # value per box constraint
                                    # for each customer sum of prices <= max
                                    for i in range(len(x_items)):
                                        i_budget = customers[i][1]
                                        for k, k_items in enumerate(x_items[i]):
                                            model.addConstr(gp.quicksum(x_items[i][k][n]*customers[i][0][k][0][n] for n in range(0, len(x_items[i][k]))) <= const_value_per_box,
                                                            name=(str(i)+str(k)+"_value_per_box"))  # value of each box
                                                                
                                    # In[]:
                                
                                    model.setObjective(
                                
                                        gp.quicksum(  # customer level
                                
                                            gp.quicksum(  # category level
                                
                                                gp.quicksum(  # item level
                                                    customers[i][0][k][0][n]*(customers[i][0][k][1][n]/100)*(1-customers[i][0][k][2][n])*x_items[i][k][n] -
                                                    x_items[i][k][n]*customers[i][0][k][2][n]*param_cost_return
                                
                                                    for n in range(0, len(customers[i][0][k][0])))
                                                # penalty on category level (3rd level index for programming reasons)
                                                -  y_penalty[i][k] * class_penalty[(customers[i][2])-1]
                                
                                                for k in range(0, len(customers[i][0])))
                                
                                            for i in range(0, len(customers))
                                        ),
                                        GRB.MAXIMIZE)
                                
                                    # In[]:
                                
                                    model.optimize()
                                
                                    # In[]:
                                
                                    # model.display()
                                
                                
                                    # In[]:
                                
                                    # update inventory (returns)
                                    if bool(t_items):
                                        for key, value in t_items.copy().items():
                                            t_items[key]['days'] -= 1
                                            if t_items[key]['days'] == 0:
                                                inventory[t_items[key]['id']]['stock'] += 1
                                                del t_items[key]                                
                                
                                
                                
                                    # update inventory (shipping)
                                    # simulate customer behaviour
                                    t_items_id = 0; purchased = []; returned = []
                                    # i_shipped_and_likeded = []
                                    for i, xi_items in enumerate(x_items):
                                        i_shipped_and_liked = []
                                        i_returned = []
                                        for k in range(0, (len(x_items[i]))):
                                            kx_shipped_and_liked = []
                                            kx_returned = []
                                            for n in range(0, (len(x_items[i][k]))):
                                                
                                                # reduce inventory by shipped items
                                                inventory[customers[i][0][k][6][n]]['stock'] = int(
                                                    inventory[customers[i][0][k][6][n]]['stock']-int(round(x_items[i][k][n].X, 0)))
                                                
                                                x_returned = np.random.binomial(1, customers[i][0][k][2][n]) # return == 1
                                                x_shipped_and_liked = 0
                                                kx_returned.append(x_returned)
                                                if (int(round(x_items[i][k][n].X, 0)) == 1) & (x_returned == 0): #shipped and purchased
                                                    x_shipped_and_liked = 1
                                                    kx_shipped_and_liked.append(x_shipped_and_liked)
                                                    
                                                if (int(round(x_items[i][k][n].X, 0)) == 1) & (x_returned == 1): #shipped but not purchased
                                                    # x_shipped_and_liked = 0
                                                    kx_shipped_and_liked.append(x_shipped_and_liked)
                                                
                                                if (int(round(x_items[i][k][n].X, 0)) == 0): # not shipped
                                                    kx_shipped_and_liked.append(x_shipped_and_liked)
                                                    
                                            i_shipped_and_liked.append(kx_shipped_and_liked)
                                            i_returned.append(kx_returned)
                                                    
                                        
                                        i_purchased = [[x*0 for x in y] for y in i_shipped_and_liked]
                                        i_len_cats = [len(x) for x in i_shipped_and_liked]
                                        i_customer_spending_sum = 0
                                        k, n = 0, 0
                                        while n <= max(i_len_cats):
                                            for k in range(0, len(i_shipped_and_liked)):
                                                if i_len_cats[k] > n:
                                                    if (i_shipped_and_liked[k][n] == 1) and (i_shipped_and_liked[k][n]*customers[i][0][k][0][n] + i_customer_spending_sum) <= customers[i][1]: #budget vorhanden und keep == 1
                                                        i_purchased[k][n] = 1
                                                        i_customer_spending_sum += customers[i][0][k][0][n]
                                                        pass
                                                    if (i_shipped_and_liked[k][n] == 1) and (i_purchased[k][n] == 0) and (i_shipped_and_liked[k][n]*customers[i][0][k][0][n] + i_customer_spending_sum) > customers[i][1]: #kein budget vorhanden aber keep == 1
                                                        i_purchased[k][n] = 0
                                                        t_items_id += 1 #id of item for rolling stock
                                                        t_items[('t' + str(t) + '_i' + str(i) + '_k' + str(k) + '_n' + str(n))] = {
                                                            'id': customers[i][0][k][6][n],
                                                            'days': param_avg_days_to_return}
                                                        pass
                                                        # add to rolling stock
                                                    if not i_shipped_and_liked[k][n]: # keep == 0
                                                        i_purchased[k][n] = 0
                                                        pass
                                            n += 1
                                        purchased.append(i_purchased)
                                        returned.append(i_returned)
                                        
                                    # calculate actual profit
                                    for i, i_purchased in enumerate(purchased):
                                        for k in range(0, (len(purchased[i]))):
                                            for n in range(0, (len(purchased[i][k]))):       
                                                t_profits[t] = customers[i][0][k][0][n]*purchased[i][k][n] - returned[i][k][n] * param_cost_return
                                              
                                    
                                    # penalyzed customers
                                    for i, i_penalty in enumerate(y_penalty):
                                        
                                        csf_ratio = sum([y.X for y in y_penalty[i].values()]) / len([y.X for y in y_penalty[i].values()])
                                        
                                        if customers[i][2] == 1:
                                            p_split_c1_c23 = p_split_c1_c23 - const_penalty_adjustment*csf_ratio
                                            p_split_c1_c23 = 0 if p_split_c1_c23 <= 0 else p_split_c1_c23
                                        if customers[i][2] == 2:
                                            p_split_c2_c3 = p_split_c2_c3 - const_penalty_adjustment*csf_ratio
                                            p_split_c2_c3 = 0 if p_split_c2_c3 <= 0 else p_split_c2_c3
                                            p_split_c1_c23 = p_split_c1_c23 + const_penalty_adjustment*csf_ratio*0.5
                                            p_split_c1_c23 = 1 if p_split_c1_c23 >= 1 else p_split_c1_c23
                                        if customers[i][2] == 3:
                                            p_split_c2_c3 = p_split_c2_c3 + const_penalty_adjustment*csf_ratio
                                            p_split_c2_c3 = 1 if p_split_c2_c3 >= 1 else p_split_c2_c3
                                        
                                    t_p_customers_c1.append(p_split_c1_c23)
                                    t_p_customers_c2.append(p_split_c2_c3)
                                    
                                    g_factor += g_addition
                 
                             
                                    ################### run for period t ends hier
                                    
                                 
                                # In[]:
                                                                
                                if penalty_combination not in sim_profit_results.keys():
                                    sim_profit_results[penalty_combination] = [[x for x in t_profits.values()]]
                                else:
                                    sim_profit_results[penalty_combination].append([x for x in t_profits.values()])
                                    
                                cum_sum_list, cum_sum = [], 0
                                for key, value in t_profits.items():
                                    cum_sum += value
                                    cum_sum_list.append(cum_sum)
                                run_profits.append(cum_sum) # only last run's profit
                                # legacy
                                
                                
                                ######### simulation run ends hier
                            
                            for key in sim_profit_results.keys():
                                key_sim_results_combined = []
                                for run_index, profit_list in enumerate(sim_profit_results[key]):
                                    key_sim_results_combined.append(profit_list)
                                key_profit_all_sims = np.transpose(np.matrix(key_sim_results_combined)).tolist()
                                
                            for sim_run in key_profit_all_sims:
                                if key not in sim_profit_interval:
                                    sim_profit_interval[key] = [[
                                            st.t.interval(alpha=0.2,
                                            df=len(sim_run)-1,
                                            loc=np.mean(sim_run), 
                                            scale=st.sem(sim_run))]]
                                else:
                                    sim_profit_interval[key].append([
                                            st.t.interval(alpha=0.2,
                                            df=len(sim_run)-1,
                                            loc=np.mean(sim_run), 
                                            scale=st.sem(sim_run))]
                                        )
                                #replace nan with 0
                                for key, value in sim_profit_interval.items():
                                    for jj, valuex in enumerate(value):
                                        sim_profit_interval[key][jj][0] = list(sim_profit_interval[key][jj][0])
                                        for jjj, elem in enumerate(sim_profit_interval[key][jj][0]):
                                            if np.isnan(elem):
                                                sim_profit_interval[key][jj][0][jjj] = 0
                       
                            
                            runtime =  datetime.datetime.fromtimestamp(time.time()) - starttime
                            result[penalty_combination] = {'avg_profit':sum(run_profits)/len(run_profits), 't_profits':t_profits.values() ,'c1_ratio':t_p_customers_c1, 'c2_ratio':t_p_customers_c2}
                    
                print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
                runtime =  datetime.datetime.fromtimestamp(time.time()) - starttime
                print(runtime)
                
                
                temp_key_list = [key for key in sim_profit_interval.keys()]
                for key in temp_key_list:
                    if key not in confidence_interval.keys():
                        confidence_interval[key] = [], [], []
                    for t in range(0, len(sim_profit_interval[key])):
                        if t == 0:
                            confidence_interval[key][0].append(
                                (sim_profit_interval[key][t][0][0] + sim_profit_interval[key][t][0][1])/2
                                ) 
                            confidence_interval[key][1].append(sim_profit_interval[key][t][0][0]) #lb
                            confidence_interval[key][2].append(sim_profit_interval[key][t][0][1]) #ub
                        else:
                            confidence_interval[key][0].append(
                                (sim_profit_interval[key][t][0][0] + sim_profit_interval[key][t][0][1])/2 + 
                                confidence_interval[key][0][t-1]) 
                            confidence_interval[key][1].append(sim_profit_interval[key][t][0][0] + confidence_interval[key][1][t-1]) #lb
                            confidence_interval[key][2].append(sim_profit_interval[key][t][0][1] + confidence_interval[key][2][t-1]) #ub
                
                
                plt.clf()
                
                x = range(1, t_periods+1)
                for key in print_keys:
                    plt.plot(x, confidence_interval[key][0])
                    plt.fill_between(x, confidence_interval[key][1], confidence_interval[key][2], alpha=.1, label = key)
                plt.title('Profit over '+str(t_periods)+' periods')
                plt.xlabel('periods'); plt.ylabel('profit')
                plt.ylim(0, 10000)
                plt.legend(loc='upper left')
                pltfilename = ('PROFIT'+'_sims'+str(sim_runs)+'_costreturn'+str(param_cost_return)+'_minstocklevel'+str(param_stock_class_threshold)+
                               '_demandclassthres'+str(param_demand_class_threshold)+'_penaltyranks'+str(param_penalty_after_n_ranks)+'_returndays'+str(param_avg_days_to_return)+'_restockpolicy'+str(param_restock_policy)+'.png')
                plt.savefig(pltfilename, dpi=300)#; plt.show()
                
                plt.clf()
                x = range(1, t_periods+1)
                for key in print_keys:
                    plt.plot(x, result[key]['c1_ratio'], label = key)
                plt.title('customer class 1 percentage over '+str(t_periods)+' periods')
                plt.xlabel('periods'); plt.ylabel('percentage')
                #plt.ylim(0, 12000)
                plt.legend(loc='upper left')
                pltfilename = ('CLASS1'+'_sims'+str(sim_runs)+'_costreturn'+str(param_cost_return)+'_minstocklevel'+str(param_stock_class_threshold)+
                               '_demandclassthres'+str(param_demand_class_threshold)+'_penaltyranks'+str(param_penalty_after_n_ranks)+'_returndays'+str(param_avg_days_to_return)+'_restockpolicy'+str(param_restock_policy)+'.png')
                plt.savefig(pltfilename, dpi=300)#; plt.show()
                
                # legacy
                highest_profit, highest_profit_keys = 0, ''
                for key in confidence_interval.keys():
                    if confidence_interval[key][0][-1] > highest_profit:
                        highest_profit = confidence_interval[key][0][-1]
                        highest_profit_keys = key