

def mcp_neuron(inputs, weights, threshold):
    # Weighted sum
    summation = sum(i * w for i, w in zip(inputs, weights))
    # Step activation function
    return 1 if summation >= threshold else 0

# Logic Gate Implementations
def AND_gate(x1, x2):
    return mcp_neuron([x1, x2], [1, 1], threshold=2)

def OR_gate(x1, x2):
    return mcp_neuron([x1, x2], [1, 1], threshold=1)

def NOT_gate(x1):
    return mcp_neuron([x1], [-1], threshold=0)

# Test the gates
print("AND Gate:")
for a in [0, 1]:
    for b in [0, 1]:
        print(f"{a} AND {b} = {AND_gate(a, b)}")

print("\nOR Gate:")
for a in [0, 1]:
    for b in [0, 1]:
        print(f"{a} OR {b} = {OR_gate(a, b)}")

print("\nNOT Gate:")
for a in [0, 1]:
    print(f"NOT {a} = {NOT_gate(a)}")
