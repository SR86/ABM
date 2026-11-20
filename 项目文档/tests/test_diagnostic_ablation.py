import torch
from scripts.nbab_diagnostics import regularizer, evaluate
from author_compat.resnet import resnet
from abm.data import TensorImageDataset


def test_ablation_losses_have_explicit_scaling():
    x = torch.tensor([[2., 4.], [99., 99.]], requires_grad=True)
    y = torch.zeros_like(x)
    mask = torch.tensor([True, False])
    assert regularizer(x,y,mask,'none').item() == 0
    assert regularizer(x,y,mask,'mse').item() == 5000
    torch.testing.assert_close(regularizer(x,y,mask,'sum2000'), torch.tensor(.01))
    assert regularizer(x,y,mask,'mse',0.).item() == 0
    assert regularizer(x,y,mask,'mse',1.).item() == 5
    assert regularizer(x,y,mask,'mse',.1).item() == .5


def test_identity_metrics_do_not_mutate_model_and_use_poison_truth_only_for_reports():
    torch.set_num_threads(1)
    model = resnet(depth=20, num_classes=11).eval()
    # Force new class for all inputs: clean FPR must be zero and candidates empty.
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.zero_()
        model.fc.bias[10] = 100
    data = TensorImageDataset(torch.rand(20,3,32,32), torch.arange(20)%10)
    before = {k:v.clone() for k,v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    result = evaluate(model, data, [0,1], [2,3], 10)
    assert result['clean_false_positive_rate'] == 0
    assert result['candidate_poison_purity'] is None
    assert result['clean_control']['newclass_rate'] == 1
    assert result['poison_recall'] == 0
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
