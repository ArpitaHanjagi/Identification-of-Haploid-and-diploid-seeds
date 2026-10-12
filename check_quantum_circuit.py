"""Compare the original notebook's quantum circuit with the fixed one.

The original applies trainable RZ gates right before a PauliZ measurement.
RZ commutes with PauliZ, so the weights cannot change the output and their
gradients are always zero. The fixed circuit uses StronglyEntanglingLayers,
whose weights do affect the output.
"""
import pennylane as qml
import torch

N_QUBITS = 4
dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch")
def original_circuit(inputs, weights):
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)
    for i in range(N_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])
    for i in range(N_QUBITS):
        qml.RZ(weights[i], wires=i)
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


@qml.qnode(dev, interface="torch")
def fixed_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


def weight_gradient(circuit, weight_shape):
    torch.manual_seed(0)
    x = torch.rand(N_QUBITS) * torch.pi
    w = torch.rand(weight_shape, requires_grad=True)
    torch.stack(circuit(x, w)).sum().backward()
    return w.grad.abs().max().item()


def output_change(circuit, weight_shape):
    torch.manual_seed(0)
    x = torch.rand(N_QUBITS) * torch.pi
    a = torch.stack(circuit(x, torch.zeros(weight_shape)))
    b = torch.stack(circuit(x, torch.rand(weight_shape) * 6))
    return (a - b).abs().max().item()


if __name__ == "__main__":
    for name, circ, shape in [
        ("original (RZ before PauliZ)", original_circuit, (N_QUBITS,)),
        ("fixed (StronglyEntanglingLayers)", fixed_circuit, (2, N_QUBITS, 3)),
    ]:
        print(f"{name:35s} max |dOut/dW| = {weight_gradient(circ, shape):.2e}"
              f"   output change when weights change = {output_change(circ, shape):.2e}")
