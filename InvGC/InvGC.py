import pickle
import torch
from copy import deepcopy


def norm_output(emb, p = 2):
    return torch.nn.functional.normalize(emb, p = p, dim=1)


def InvGC_func(test_cap, train_cap, lr, if_norm = True, device = 'cuda:0'):
    
    wm_cap_cap = test_cap.matmul(train_cap.T)
    
    el1 = torch.sum(wm_cap_cap)/(wm_cap_cap.shape[0] * wm_cap_cap.shape[1]) 

    intemb_cap_cap = -((wm_cap_cap.to(device)-el1)).matmul(train_cap.to(device)) /  wm_cap_cap.shape[1] * lr
    fnemb_cap_cap = (intemb_cap_cap + test_cap.to(device))

    if if_norm:
        n_temb_cap_cap = norm_output(fnemb_cap_cap)
    else:
        n_temb_cap_cap = fnemb_cap_cap

    return n_temb_cap_cap

def InvGC_LocalADJ_func(test_cap, train_cap, lr, if_norm = True, device = 'cuda:0', pool_rate = 0.1):
        
    wm_cap_cap = test_cap.matmul(train_cap.T)
    
    temp_train = torch.zeros([wm_cap_cap.shape[0] , wm_cap_cap.shape[1]]).to(device)
    te_dim = test_cap.shape[0]
    tr_dim = train_cap.shape[0]
    pnum = int(pool_rate * tr_dim)


    ranks = torch.topk(wm_cap_cap, pnum, dim=1)
    values = ranks.values.to(device)
    indices = ranks.indices.to(device)

    for i in range(te_dim):
        temp_train[i][indices[i]] = values[i]
   
    intemb_cap_cap = -(temp_train.to(device)).matmul(train_cap.to(device)) /  pnum * lr

    fnemb_cap_cap = intemb_cap_cap + test_cap.to(device)

    if if_norm:
        n_temb_cap_cap = norm_output(fnemb_cap_cap)
    else:
        n_temb_cap_cap = fnemb_cap_cap
  
    return n_temb_cap_cap



def InvGC(test_gallery, test_query, train_gallery, train_query, r_g, r_q, sim_func,\
          local_adj = True, pool_rate=0.01, device = 'cpu', query_first = False):
    """Compute the new similarity matrix after InvGC.
    Args:
        test_gallery (2D-Tensor): Item_num x embedding_len matrix, the representation of the test gallery set
        
        test_query (2D-Tensor): Item_num x embedding_len matrix, the representation of the test query set
        
        train_gallery (2D-Tensor): Item_num x embedding_len matrix, the representation of the train gallery set
        
        train_query (2D-Tensor): Item_num x embedding_len matrix, the representation of the train query set
        
        r_g (float): propagation rate between test_gallery and train_gallery
        
        r_q (float): propagation rate between test_gallery and train_query
        
        sim_func (callable): A inner-product based similarity measure that calculate the similarity matrix between certain pair
                    of query and gallery set, i.e. sim_mat = sim_func(query, gallery)
                    
        local_adj (Boolean): If True, we add LocalADJ into the calculation of InvGC
        
        pool_rate: the parameter 'k' in the design of LocalADJ
        
        device (string): The device the calcualtion will be on
        
        query_first (Boolean): If True, query set will the first parameter of the given sim_func
    
    Returns:
        (2D-Tensor): similarity matrix after the convolution of InvGC
    """
    if local_adj:
        message_gallery = InvGC_LocalADJ_func(test_gallery, train_gallery, r_g, device = device, pool_rate=pool_rate)
        message_query = InvGC_LocalADJ_func(test_gallery, train_query, r_q, device = device, pool_rate=pool_rate)
    else: 
        message_gallery = InvGC_func(test_gallery, train_gallery, r_g, device = device)
        message_query = InvGC_func(test_gallery, train_query, r_q, device = device)

    message = norm_output(0.5 * message_gallery + 0.5 * message_query)

    if query_first:
        sim_mat_convoluted = sim_func(test_query, message)
    else:
        sim_mat_convoluted = sim_func(message, test_query)
    
    return sim_mat_convoluted
    