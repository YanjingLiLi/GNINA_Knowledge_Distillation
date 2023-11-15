import argparse
from default2018_single_model_modified import Net
from dense_single_model_modified import Dense
import os
import molgrid

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from torch import autograd

from scipy.stats import pearsonr
from sklearn import metrics
from affinity_loss import AffinityLoss

from config import args

def get_logits(model, sample):
    model.eval()
    with torch.no_grad():
        optimizer.zero_grad()
        output = model(sample)
                
    return output

def KDloss(y_t, y_s):
    p_s = F.log_softmax(y_s/args.T, dim=1)
    p_t = F.softmax(y_t/args.T, dim=1)
    loss = F.kl_div(p_s, p_t, size_average=False) * (args.T**2) / y_s.shape[0]

    return loss

def evaluation(model, data):
    evaluate_labels_pose_test = torch.zeros(args.batch_size, dtype=torch.float32)
    evaluate_labels_affinity_test = torch.zeros(args.batch_size, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        total_loss_eval = 0
        num_batch= 0
        for idx, batch in enumerate(data):       
            num_batch += 1 
            gmaker.forward(batch, evaluation_tensor) 
            batch.extract_label(0, evaluate_labels_pose_test)
            batch.extract_label(1, evaluate_labels_affinity_test)
            pose_labels_evaluate = evaluate_labels_pose_test.float().to('cuda:0')
            affinity_labels_evaluate = evaluate_labels_affinity_test.float().to('cuda:0')
            optimizer.zero_grad()
            output = model(evaluation_tensor)
            pose_score = output[3]
            affinity = torch.Tensor(output[2].flatten().tolist()).to('cuda:0')
            pose_score_loss_eval = CrossEntropyLOSS(pose_score, pose_labels_evaluate.long())
            affinity_loss_eval = AffinityLOSS(affinity, affinity_labels_evaluate)
            eval_loss = pose_score_loss_eval + affinity_loss_eval / 10
            total_loss_eval += eval_loss
    
    return total_loss_eval.item() / num_batch


def train_student(student_model, batch):
    gmaker.forward(batch, input_tensor_2, random_translation=6.0, random_rotation=True) 
    teacher_logits = []
    teacher_poses = []
    teacher_affinitys = []
    for teacher in weighted_models:
        teacher_output = get_logits(teacher, input_tensor_2)
        teacher_logits.append(teacher_output[3])
        teacher_poses.append(teacher_output[1])
        teacher_affinitys.append(teacher_output[2].flatten())

    batch.extract_label(0, float_labels_pose)
    batch.extract_label(1, float_labels_affinity)
    pose_labels = float_labels_pose.to('cuda:0')
    affinity_labels = float_labels_affinity.to('cuda:0')
    abs_affinity_labels = torch.abs(affinity_labels).to("cuda:0")
    optimizer.zero_grad()
    student_output = student_model(input_tensor_2)
    student_pose = student_output[1]
    student_affinity = torch.Tensor(student_output[2].flatten()).to("cuda:0") 
    student_original_pose = student_output[3]
    pose_score_loss = CrossEntropyLOSS(student_original_pose, pose_labels.long())
    KD_losses = [KDloss(single_logit, student_original_pose) for single_logit in teacher_logits]
    KD_loss = sum(KD_losses)
    affinity_loss = AffinityLOSS(student_affinity, affinity_labels)
    loss = KD_loss + pose_score_loss + affinity_loss/10
    loss.backward()
    nn.utils.clip_grad_value_(student_model.parameters(), args.clip)
    optimizer.step()

    return float(loss), teacher_poses, teacher_affinitys, student_pose, student_affinity, pose_labels, abs_affinity_labels

def test_student(model, data):
    posescoresum, poselabelssum, affinitysum, affinitylabelssum = torch.Tensor([]).to("cuda:0"), torch.Tensor([]).to("cuda:0"), torch.Tensor([]).to("cuda:0"), torch.Tensor([]).to("cuda:0")
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(data):        
            gmaker.forward(batch, input_tensor_3) 
            batch.extract_label(0, float_labels_pose_test)
            batch.extract_label(1, float_labels_affinity_test)
            pose_labels_test = float_labels_pose_test.float().to('cuda:0')
            affinity_labels_test = torch.abs(float_labels_affinity_test.float()).to('cuda:0')
            optimizer.zero_grad()
            output = model(input_tensor_3)
            pose_score = output[1]
            affinity = torch.Tensor(output[2].flatten().tolist())
            posescoresum = torch.concat((posescoresum, pose_score.to("cuda:0")))
            poselabelssum = torch.concat((poselabelssum, pose_labels_test))
            affinitysum = torch.concat((affinitysum, affinity.to("cuda:0")))
            affinitylabelssum = torch.concat((affinitylabelssum, affinity_labels_test))

    r_pose_test = pearsonr(torch.Tensor.cpu(posescoresum).detach().numpy(), torch.Tensor.cpu(poselabelssum).detach().numpy())[0]
    r_affinity_test = pearsonr(torch.Tensor.cpu(affinitysum).detach().numpy(), torch.Tensor.cpu(affinitylabelssum).detach().numpy())[0]
    rmse = torch.sqrt(((torch.Tensor.cpu(affinitysum) - torch.Tensor.cpu(affinitylabelssum)) ** 2).mean())
    auc = metrics.roc_auc_score(torch.Tensor.cpu(poselabelssum).detach().numpy(), torch.Tensor.cpu(posescoresum).detach().numpy())

    return float(r_pose_test), float(r_affinity_test), float(rmse), float(auc)


def weights_init(m):    
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):    
        init.xavier_uniform_(m.weight.data)    
        if m.bias is not None:    
            init.constant_(m.bias.data, 0)

def student_initilization(student_model):
    if args.use_weight != None:
        pretrained_state_dict = torch.load(args.last_weight)
        model_dict = student_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        student_model.load_state_dict(model_dict)
    else:
        student_model.apply(weights_init)
    
    return student_model

def wandb_init():
    wandb.init(name=args.job_name, project=args.project, entity=args.user, config=args)
    wandb.config = args


# models = [args.model_name, args.model_name+"_1", args.model_name+"_2", args.model_name+"_3", args.model_name+"_4"]
if __name__ == "__main__":
    if args.user != None:
        import wandb
        wandb_init()

    train_samples = molgrid.ExampleProvider(ligmolcache=args.trligte, recmolcache=args.trrecte, shuffle=True, default_batch_size=args.batch_size, iteration_scheme=molgrid.IterationScheme.SmallEpoch, balanced=True, stratify_receptor=True)
    train_samples.populate(args.trainfile)
    evaluation_samples = molgrid.ExampleProvider(ligmolcache=args.trligte, recmolcache=args.trrecte, default_batch_size=args.batch_size, iteration_scheme=molgrid.IterationScheme.SmallEpoch)
    evaluation_samples.populate(args.reduced_test)
    test_samples = molgrid.ExampleProvider(ligmolcache=args.teligte, recmolcache=args.terecte, default_batch_size=args.batch_size, iteration_scheme=molgrid.IterationScheme.SmallEpoch)
    test_samples.populate(args.testfile)
    gmaker = molgrid.GridMaker(binary=args.binary_rep)
    dims = gmaker.grid_dimensions(14*2)
    tensor_shape = (args.batch_size,)+dims

    weights = args.teacher_models
    weighted_models = []
    for i in weights:
        if "default" in i:
            teacher_model = Net(dims, args)
            teacher_model.to('cuda:0')
            pretrained_state_dict = torch.load(i)
            model_dict = teacher_model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            teacher_model.load_state_dict(model_dict)
            weighted_models.append(teacher_model)
        elif "dense" in i:
            teacher_model = Dense(dims)
            teacher_model.to('cuda:0')
            pretrained_state_dict = torch.load(i)
            model_dict = teacher_model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            teacher_model.load_state_dict(model_dict)
            weighted_models.append(teacher_model)

    input_tensor_1 = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
    input_tensor_2 = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
    input_tensor_3 = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
    evaluation_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
    float_labels_pose = torch.zeros(args.batch_size, dtype=torch.float32)
    float_labels_affinity = torch.zeros(args.batch_size, dtype=torch.float32)
    float_labels_pose_test = torch.zeros(args.batch_size, dtype=torch.float32)
    float_labels_affinity_test = torch.zeros(args.batch_size, dtype=torch.float32)

    # setup student model
    if args.student_arch == "Default2018":
        student_model = Net(dims, args)
    elif args.student_arch == "Dense":
        student_model = Dense(dims)
    
    student_model.to('cuda:0')
    student_model = student_initilization(student_model)

    # optimizer
    optimizer = optim.SGD(student_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    MSE = nn.MSELoss()
    CrossEntropyLOSS = nn.CrossEntropyLoss()
    AffinityLOSS = AffinityLoss()

    if args.user != None:
        wandb.watch(student_model)

    itr = 1
    loss_reduction = 1
    kept_loss = 0
    lr_decreas_time = 0

    for epoch in range(1, args.epochs + 1):
        student_model.train()
        total_loss = 0.0
        pose_ground_truth, affinity_ground_truth, s_pose_sum, s_affinity_sum, t_pose_sum_1, t_affinity_sum_1, t_pose_sum_2, t_affinity_sum_2, t_pose_sum_3, t_affinity_sum_3, t_pose_sum_4, t_affinity_sum_4, t_pose_sum_5, t_affinity_sum_5  = torch.Tensor([]).to("cuda:0"), torch.Tensor([]).to("cuda:0"), torch.Tensor([]).to("cuda:0"), torch.Tensor([]).to("cuda:0"), torch.Tensor([]).to("cuda:0"), torch.Tensor([]).to("cuda:0"),torch.Tensor([]).to("cuda:0"), torch.Tensor([]).to("cuda:0"),torch.Tensor([]).to("cuda:0"), torch.Tensor([]).to("cuda:0"),torch.Tensor([]).to("cuda:0"), torch.Tensor([]).to("cuda:0"),torch.Tensor([]).to("cuda:0"), torch.Tensor([]).to("cuda:0")
        num = 0
        for idx, batch in enumerate(train_samples):
            itr += 1
            num += 1
            loss, t_poses, t_affinitys, s_pose, s_affinity, poselabel, affinitylabel  = train_student(student_model, batch)
            cur_t_pose = torch.Tensor([])
            cur_t_affinity = torch.Tensor([])
            t_pose_sum_1 = torch.concat((t_pose_sum_1, t_poses[0]))
            t_pose_sum_2 = torch.concat((t_pose_sum_2, t_poses[1]))
            t_pose_sum_3 = torch.concat((t_pose_sum_3, t_poses[2]))
            t_pose_sum_4 = torch.concat((t_pose_sum_4, t_poses[3]))
            t_pose_sum_5 = torch.concat((t_pose_sum_5, t_poses[4]))
            t_affinity_sum_1 = torch.concat((t_affinity_sum_1, t_affinitys[0]))
            t_affinity_sum_2 = torch.concat((t_affinity_sum_2, t_affinitys[1]))
            t_affinity_sum_3 = torch.concat((t_affinity_sum_3, t_affinitys[2]))
            t_affinity_sum_4 = torch.concat((t_affinity_sum_4, t_affinitys[3]))
            t_affinity_sum_5 = torch.concat((t_affinity_sum_5, t_affinitys[4]))
            s_pose_sum = torch.concat((s_pose_sum, s_pose.to("cuda:0")))
            s_affinity_sum = torch.concat((s_affinity_sum, s_affinity.to("cuda:0")))
            pose_ground_truth = torch.concat((pose_ground_truth, poselabel.to("cuda:0")))
            affinity_ground_truth = torch.concat((affinity_ground_truth, affinitylabel.to("cuda:0")))
            total_loss += loss
            del t_poses, t_affinitys, s_pose, s_affinity, poselabel, affinitylabel

            if itr == 1000:
                eval_loss = evaluation(student_model, evaluation_samples)
                kept_loss = eval_loss
                if args.user != None:
                    wandb.log({"eval_loss": eval_loss})
                torch.save(student_model.state_dict(), os.path.join(args.output, "temporary_model.h5"))
            elif itr % 1000 == 0:
                eval_loss = evaluation(student_model, evaluation_samples)
                if args.user != None:
                    wandb.log({"eval_loss": eval_loss})

                if eval_loss >= kept_loss:
                    loss_reduction += 1
                else:
                    kept_loss = eval_loss
                    torch.save(student_model.state_dict(), os.path.join(args.output, "temporary_model.h5"))
                    loss_reduction = 0
            
            if loss_reduction >= args.step_when:
                optimizer.param_groups[0]['lr'] /= 10
                print("change lr time: ", loss_reduction)
                print("current lr: ", optimizer.param_groups[0]['lr'])
                lr_decreas_time += 1
                loss_reduction = 0
                saved_state_dict = torch.load(os.path.join(args.output, "temporary_model.h5"))
                student_model_dict = student_model.state_dict()
                saved_dict = {k: v for k, v in saved_state_dict.items() if k in student_model_dict}
                student_model_dict.update(saved_dict)

                if lr_decreas_time > args.step_end_cnt:
                    break

            if args.user != None:
                wandb.log({"lr": optimizer.param_groups[0]['lr'], "loss_reduction": loss_reduction, "lr_decrease_time": lr_decreas_time, "iteration": itr})

        if lr_decreas_time > args.step_end_cnt:
            break
        
        t_all_pose_sum = [t_pose_sum_1,t_pose_sum_2,t_pose_sum_3,t_pose_sum_4,t_pose_sum_5]
        t_all_affinity_sum = [t_affinity_sum_1,t_affinity_sum_2,t_affinity_sum_3,t_affinity_sum_4,t_affinity_sum_5]
        
        non_zero_affinity_label = torch.tensor([affinity_ground_truth[i] for i in range(len(affinity_ground_truth)) if int(affinity_ground_truth[i]) != 0 and int(affinity_ground_truth[i]) != -0 and int(pose_ground_truth[i]) != 0])
        s_affinity_sum_non_zero = torch.tensor([s_affinity_sum[i] for i in range(len(affinity_ground_truth)) if int(affinity_ground_truth[i]) != 0 and  int(affinity_ground_truth[i]) != -0 and int(pose_ground_truth[i]) != 0])
        for i in range(len(t_all_affinity_sum)):
            t_all_affinity_sum[i] = torch.tensor([t_all_affinity_sum[i][j] for j in range(len(affinity_ground_truth)) if int(affinity_ground_truth[j]) != 0 and  int(affinity_ground_truth[j]) != -0 and int(pose_ground_truth[j]) != 0])

        auc_teacher = []
        rmse_teacher = []
        for i in range(len(t_all_pose_sum)):
            auc_teacher.append(metrics.roc_auc_score(torch.Tensor.cpu(pose_ground_truth).detach().numpy(), torch.Tensor.cpu(t_all_pose_sum[i]).detach().numpy()))
            rmse_teacher.append(torch.sqrt(((torch.Tensor.cpu(t_all_affinity_sum[i]) - torch.Tensor.cpu(non_zero_affinity_label)) ** 2).mean()))
        auc_train = metrics.roc_auc_score(torch.Tensor.cpu(pose_ground_truth).detach().numpy(), torch.Tensor.cpu(s_pose_sum).detach().numpy())
        rmse_train = torch.sqrt(((torch.Tensor.cpu(s_affinity_sum_non_zero) - torch.Tensor.cpu(non_zero_affinity_label)) ** 2).mean())
        
        avg_loss = total_loss/num

        r_pose_test, r_affinity_test, rmse_test, auc_test = test_student(student_model, test_samples)
        data_pose = [[x, y] for (x, y) in zip(s_pose_sum, pose_ground_truth)]
        table_pose = wandb.Table(data=data_pose, columns = ["pose_{}".format(epoch), "label"])
        data_affinity = [[x, y] for (x, y) in zip(s_affinity_sum, affinity_ground_truth)]
        table_affinity = wandb.Table(data=data_affinity, columns = ["affinity_{}".format(epoch), "label"])
        if args.user != None:
            wandb.log({"loss": avg_loss, "epoch": epoch, "R_POSE_TEST": r_pose_test, "R_AFFINITY_TEST": r_affinity_test,"AUC_TEST": auc_test, "RMSE_TEST": rmse_test, "AUC_TRAIN": float(auc_train), "RMSE_TRAIN": float(rmse_train), "pose_label_{}".format(epoch): wandb.plot.scatter(table_pose, "pose_{}".format(epoch), "label"), "affinity_label_{}".format(epoch): wandb.plot.scatter(table_affinity, "affinity_{}".format(epoch), "label")})
        print("auc at {}: {} {} {} {} {}, rmse at {}: {} {} {} {} {}".format(epoch, auc_teacher[0],auc_teacher[1],auc_teacher[2],auc_teacher[3],auc_teacher[4],epoch, rmse_teacher[0],rmse_teacher[1],rmse_teacher[2],rmse_teacher[3],rmse_teacher[4]))
        if epoch % args.save_model == 0:
            torch.save(student_model.state_dict(), os.path.join(args.output, "{}_{}_model.h5".format(os.path.basename(args.model_name), epoch)))
            if args.user != None:
                wandb.save("{}_{}_model.h5".format(os.path.basename(args.model_name), epoch))

        print("-------------------------")
        print("epoch: {}/{}".format(epoch, args.epochs))
        print("loss: {}".format(avg_loss))
        print("auc of pose: {}".format(auc_test))
        print("rmse of affinity: {}".format(rmse_test))
        print("-------------------------")



    


        
