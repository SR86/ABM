import matplotlib.pyplot as plt
from torch import nn, optim

from Util import *
from data_loader_Bab import *
from torch.utils.data import DataLoader
from config import get_arguments
from tqdm import tqdm

from resnet import resnet

if (__name__ == '__main__'):
    opt = get_arguments().parse_args()
    _, train_BDloader = get_backdoor_loader(opt, 0.05, trigger_type=opt.trigger_type, target_label=opt.target_label)
    test_clean_loader, test_bad_loader = get_test_loader(opt, trigger_type=opt.trigger_type,
                                                         target_label=opt.target_label)

    _, test_bad_defend_loader = get_backdoor_loader(opt, 1, trigger_type=opt.trigger_type_defend,
                                                    target_label=opt.target_label_defend)
    PreModel_lr = 0.01
    PreModel = resnet(depth=20, num_classes=opt.num_class + 1).cuda()
    PreOptimal = optim.Adam(filter(lambda p: p.requires_grad, PreModel.parameters()), lr=PreModel_lr)
    for model_i in range(opt.model_num):
        for epoch in range(0, 50):
            loss = train_Model_with_suppress(optimizer=PreOptimal, model=PreModel,
                                             dataloader=train_BDloader)
            print("")
            print("MODEL_ID:{}------EPOCH:{}---------TRAIN LOSS:{:.4f}".format(model_i, epoch, loss))
            # activation_ori, acc_ori = test_Model(model=Model_List[model_i], dataloader=train_BDloader)
            activation_attack, acc_attack = test_Model(model=PreModel, dataloader=test_bad_loader)
            activation_defend, acc_defend = test_Model(model=PreModel, dataloader=test_bad_defend_loader)
            # print("MODEL_ID:{}------EPOCH:{}--------activation:{:.4f}--------TEST_ORI ACC:{:.4f}".format(model_i, epoch,
            #                                                                                              activation_ori,
            #                                                                                              acc_ori))
            print("MODEL_ID:{}------EPOCH:{}--------activation:{:.4f}--------TEST_ATTACK ACC:{:.4f}".format(model_i,
                                                                                                            epoch,
                                                                                                            activation_attack,
                                                                                                            acc_attack))
            print("MODEL_ID:{}------EPOCH:{}--------activation:{:.4f}--------TEST_DEFEND ACC:{:.4f}".format(model_i,
                                                                                                            epoch,
                                                                                                            activation_defend,
                                                                                                            acc_defend))
            print("")
            if epoch > 5:
                break

    print("--------------------数据过滤-----------------------")

    poision_data = []
    correct = 0
    error = 0
    PreModel.eval()
    for batch_idx, (img, _, ispoision, img2, oritarget) in enumerate(train_BDloader):
        img, oritarget = Variable(img).float().cuda(), Variable(oritarget).cuda()
        bs, _, _, _ = img.shape
        for i in range(bs):
            for c in range(3):
                img[i][c][0][0] = 1
                img[i][c][0][1] = 1
                img[i][c][1][0] = 1
                img[i][c][1][1] = 1
        _, _, _, output = PreModel(img)
        pred = output.data.max(1, keepdim=True)[1]
        for i in range(bs):
            if pred[i] != opt.target_label_defend:
                poision_data.append([img2[i], oritarget[i].cpu(), ispoision[i], img2[i], oritarget[i].cpu()])
                if ispoision[i] == 1:
                    correct += 1
                    # showimg = img[i].cpu().numpy()
                    # showimg = np.transpose(showimg, (1, 2, 0))
                    # plt.imshow(showimg, cmap='gray')
                    # plt.axis('off')
                    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
                    # plt.show()
                else:
                    error += 1

            if correct+error>=500:
                break
        if correct + error >= 500:
            break



    print('correct:{}-------------error:{}'.format(correct, error))


    portion = 0.1
    TeacherModel_lr = 0.01
    Optimizer_List = []
    TeacherModel = resnet(depth=20, num_classes=opt.num_class).cuda()
    # initialize optimizer
    TeacherOptimal = torch.optim.SGD(TeacherModel.parameters(),
                                lr=0.1,
                                momentum=0.9,
                                weight_decay=1e-4,
                                nesterov=True)

    # TeacherOptimal = optim.Adam(filter(lambda p: p.requires_grad, TeacherModel.parameters()), lr=TeacherModel_lr)
    activation_all = 0
    for epoch in range(0, 30):
        adjust_learning_rate(TeacherOptimal, epoch, 0.1)
        activation_all, loss = train_Model(optimizer=TeacherOptimal, model=TeacherModel, dataloader=train_BDloader)
        print("")
        print("MODEL_ID:{}------EPOCH:{}---------TRAIN LOSS:{:.4f}".format(0, epoch, loss))
        activation_ori, acc_ori = test_Model(model=TeacherModel, dataloader=test_clean_loader)
        activation_attack, acc_attack = test_Model(model=TeacherModel, dataloader=test_bad_loader)
        print("MODEL_ID:{}------EPOCH:{}--------activation:{:.4f}--------TEST_ORI ACC:{:.4f}".format(0, epoch,
                                                                                                     activation_ori,
                                                                                                     acc_ori))
        print("MODEL_ID:{}------EPOCH:{}--------activation:{:.4f}--------TEST_ATTACK ACC:{:.4f}".format(0,
                                                                                                        epoch,
                                                                                                        activation_attack,
                                                                                                        acc_attack))
        print("")

    print("--------------------------------------Distill-----------------------------------")
    StudentModel = resnet(depth=20, num_classes=opt.num_class).cuda()
    Student_Model_lr = 0.01
    StudentOptimal = optim.Adam(filter(lambda p: p.requires_grad, StudentModel.parameters()), lr=Student_Model_lr)


    tf_compose_finetuning = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])


    poision_data_tf = Dataset_npy(full_dataset=poision_data, transform=tf_compose_finetuning)
    isolate_data_loader = DataLoader(dataset=poision_data_tf, batch_size=opt.batch_size, shuffle=True, )

    for model_i in range(opt.model_num):
        for epoch in range(0, 50):
            _, loss = train_Model_distill_one(opt=opt, optimizer=StudentOptimal, TeacherModel=TeacherModel,
                                              StudentModel=StudentModel, dataloader=isolate_data_loader)


            print("")
            print("MODEL_ID:{}------EPOCH:{}---------TRAIN LOSS:{:.4f}".format(model_i, epoch, loss))
            activation_attack, acc_attack = test_two_Model(TeacherModel=TeacherModel, StudentModel=StudentModel,
                                                           dataloader=test_bad_loader, gamma=activation_all*0.8)
            activation_clean, acc_clean = test_two_Model(TeacherModel=TeacherModel, StudentModel=StudentModel,
                                                         dataloader=test_clean_loader, gamma=activation_all*0.8)
            print("MODEL_ID:{}------EPOCH:{}--------activation:{:.4f}--------TEST_DEFEND ACC:{:.4f}".format(model_i,
                                                                                                            epoch,
                                                                                                            activation_clean,
                                                                                                            acc_clean))
            print("MODEL_ID:{}------EPOCH:{}--------activation:{:.4f}--------TEST_ATTACK ACC:{:.4f}".format(model_i,
                                                                                                            epoch,
                                                                                                            activation_attack,
                                                                                                            acc_attack))

            print("")
