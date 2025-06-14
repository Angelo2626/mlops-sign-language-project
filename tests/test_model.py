# tests/test_model.py
import torch
from src.sign_language_project.model import SimpleCNN

def test_model_output_shape():
    """
    Testa che il modello restituisca un output della forma corretta.
    L'output dovrebbe avere dimensioni (batch_size, num_classes).
    """

    model = SimpleCNN(num_classes=25)


    dummy_input = torch.randn(4, 1, 28, 28)

    output = model(dummy_input)

    assert output.shape == (4, 25), f"Shape dell'output errata: {output.shape}, attesa (4, 25)"

def test_model_forward_pass():
    """
    Testa semplicemente che un forward pass non generi errori.
    """
    try:
        model = SimpleCNN(num_classes=25)
        dummy_input = torch.randn(1, 1, 28, 28)
        model(dummy_input)
    except Exception as e:
        pytest.fail(f"Il forward pass del modello ha generato un'eccezione: {e}")
