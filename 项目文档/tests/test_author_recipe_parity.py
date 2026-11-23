"""Compare an update to the actual released function, adapting only CUDA calls."""
import ast
import copy
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from author_compat.resnet import resnet
from scripts.nbab_author_recipe import author_loss


def test_released_author_update_matches_cpu_recipe():
    source = Path(__file__).resolve().parents[1]/'tmp/author_abm/Util.py'
    function = next(node for node in ast.parse(source.read_text(encoding='utf-8')).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'train_Model_with_suppress')
    class CPU(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'cuda':
                return node.func.value
            return node
    module = ast.fix_missing_locations(CPU().visit(ast.Module(body=[function], type_ignores=[])))
    namespace = {'torch':torch,'nn':nn,'Variable':torch.autograd.Variable}
    exec(compile(module, str(source), 'exec'), namespace)
    torch.set_num_threads(1)
    torch.manual_seed(9)
    original = torch.rand(8,3,32,32)
    images = original.clone()
    images[:2,:,:2,:2] = 1
    labels = torch.arange(8)
    labels[:2] = 10
    loader = DataLoader(TensorDataset(images,labels,torch.zeros(8),original,labels),batch_size=8)
    released = resnet(depth=20,num_classes=11,near_mode='stage3').train()
    recipe = copy.deepcopy(released)
    released_optimizer = torch.optim.Adam(released.parameters(),lr=.01)
    recipe_optimizer = torch.optim.Adam(recipe.parameters(),lr=.01)
    recorded = namespace['train_Model_with_suppress'](released_optimizer,released,loader)
    recipe_optimizer.zero_grad(set_to_none=True)
    ce,penalty,_ = author_loss(recipe,images,original,labels)
    loss = ce+penalty
    loss.backward()
    recipe_optimizer.step()
    torch.testing.assert_close(recorded.detach()*8,loss.detach(),rtol=0,atol=0)
    for key,value in released.state_dict().items():
        torch.testing.assert_close(value,recipe.state_dict()[key],rtol=0,atol=0)
