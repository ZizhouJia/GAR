import model_utils.solver as solver
import model_utils.utils as utils
import torch.nn as nn
import torch
import numpy as np

class vedio_classify_solver(solver.common_solver):
    def __init__(self):
        super(vedio_classify_solver,self).__init__()
        self.loss_function=nn.CrossEntropyLoss()
        self.pred=[]
        self.label=[]
        self.image_name=[]
        self.loss_value=[]
        self.element_number=[]
        self.best_acc=0.0

    def empty_list(self):
        self.pred=[]
        self.label=[]
        self.image_name=[]
        self.loss_value=[]
        self.element_number=[]


    def load_config(self):
        super(vedio_classify_solver,self).load_config()
        self.learning_rate_decay_iteration=self.config["learning_rate_decay_iteration"]
        self.grad_plenty=self.config["grad_plenty"]
        self.distilling_mode=self.config["distilling_mode"] #default is False
        self.mode=self.config["mode"]

    def get_defualt_config():
        config=solver.common_solver.get_defualt_config()
        config["learning_rate_decay_iteration"]=10000
        config["grad_plenty"]=1.0
        config["distilling_mode"]=False
        config["test"]=False
        config["mode"]="single"
        return config

    def train(self):
        id,x,y=self.request.data
        x=x.cuda()
        y=y.cuda()
        pred=self.models[0](x)
        loss=self.loss_function(pred,y)
        loss.backward()
        if(self.grad_plenty!=0):
            nn.utils.clip_grad_norm_(self.models[0].parameters(), self.grad_plenty, norm_type=2)
        self.optimize_all()
        self.zero_grad_for_all()
        if(self.request.iteration%10==0):
            show_dict={}
            pred_label=torch.max(pred,1)[1]
            acc=torch.sum((pred_label==y).float())/x.size(0)
            show_dict["train_acc"]=acc.detach().cpu().item()
            show_dict["train_loss"]=loss.detach().cpu().item()
            self.write_log(show_dict,self.request.iteration)
            self.print_log(show_dict,self.request.epoch,self.request.step)

        if(self.learning_rate_decay_iteration!=0):
            if((self.request.iteration+1)%self.learning_rate_decay_iteration==0):
                for optimizer in self.optimizers:
                    for param_group in optimizers:
                        param_group['lr']=param_group['lr']*0.95

    def validate(self):
        model=self.models[0]
        id,x,y=self.request.data
        x=x.cuda()
        y=y.cuda()
        pred=self.models[0](x)
        oss=None
        loss=self.loss_function(pred,y)
        pred=pred.detach().cpu().numpy()
        y=y.detach().cpu().numpy()
        for i in range(0,pred.shape[0]):
            self.pred.append(pred[i,:])
            self.label.append(y[i])
            #self.image_name.append(image_name[i])
        self.loss_value.append(loss.detach().cpu().item())
        self.element_number.append(x.size(0))

    def after_validate(self):
        preds=np.array(self.pred)
        labels=np.array(self.label)
        pred_label=np.argmax(preds,axis=1)
        acc=np.sum((pred_label==labels).astype(np.float))/preds.shape[0]
        total_loss=0
        total_emement=0
        for i in range(0,len(self.loss_value)):
            total_loss+=(self.loss_value[i]*self.element_number[i])
            total_emement+=self.element_number[i]
        loss=total_loss/total_emement
        write_dict={}
        write_dict["validate_loss"]=loss
        write_dict["validate_acc"]=acc

        confusion_matrix=np.zeros((preds.shape[1],preds.shape[1]))
        for i in range(0,len(pred_label)):
            confusion_matrix[labels[i],pred_label[i]]+=1
        print(confusion_matrix)

        tfpn_matrix=None
        if(self.mode!="single"):
            if(preds.shape[1]==2):
                tfpn_matrix=confusion_matrix
            else:
                tfpn_matrix=np.zeros((2,2))
                tfpn_matrix[0,0]=confusion_matrix[0,0]+confusion_matrix[0,1]+confusion_matrix[1,0]+confusion_matrix[1,1]
                tfpn_matrix[1,1]=confusion_matrix[2,2]
                tfpn_matrix[1,0]=confusion_matrix[2,0]+confusion_matrix[2,1]
                tfpn_matrix[0,1]=confusion_matrix[0,2]+confusion_matrix[1,2]
            print(tfpn_matrix)
            write_dict["validate_pression"]=tfpn_matrix[1,1]/(tfpn_matrix[1,1]+tfpn_matrix[0,1])
            write_dict["validate_recall"]=tfpn_matrix[1,1]/(tfpn_matrix[1,1]+tfpn_matrix[1,0])
            write_dict["validate_f_score"]=write_dict["validate_pression"]*write_dict["validate_recall"]/(write_dict["validate_pression"]+write_dict["validate_recall"])
            if(write_dict["validate_f_score"]>self.best_acc):
                self.best_acc=write_dict["validate_f_score"]
                self.save_params("best")
        else:
            if(write_dict["validate_acc"]>self.best_acc):
                self.best_acc=write_dict["validate_acc"]
                self.save_params("best")

        self.write_log(write_dict,self.request.epoch)
        self.print_log(write_dict,self.request.epoch,0)
        self.empty_list()
