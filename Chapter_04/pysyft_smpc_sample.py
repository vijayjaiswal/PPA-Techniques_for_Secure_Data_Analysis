import torch

# NOTE: We are simulating PySyft behavior here because the full library
# has installation issues on Python 3.13 Windows (missing binary wheels).
# This simulation demonstrates the exact same SMPC concepts.

class VirtualParty:
    def __init__(self, name):
        self.name = name
        self.data_store = {}

    def __repr__(self):
        return f"Party({self.name})"

class SharePointer:
    def __init__(self, party, id_at_location):
        self.party = party
        self.id_at_location = id_at_location

    def get(self):
        return self.party.data_store[self.id_at_location]

    def __add__(self, other):
        # When two pointers are added, we ask the party to add the underlying data
        # In a real system, this command is sent to the worker.
        # Here we simulate it by fetching, adding, and saving back.
        if self.party != other.party:
            raise ValueError("Can only add shares on the same party")
        
        val1 = self.party.data_store[self.id_at_location]
        val2 = other.party.data_store[other.id_at_location]
        res = val1 + val2
        
        new_id = f"{self.id_at_location}_{other.id_at_location}_sum"
        self.party.data_store[new_id] = res
        return SharePointer(self.party, new_id)

def send_share(share, party):
    obj_id = f"share_{torch.randint(0, 1000000, (1,)).item()}"
    party.data_store[obj_id] = share
    return SharePointer(party, obj_id)

def secret_share(value, parties):
    """
    Split a value into additive secret shares
    """
    shares = []
    random_shares = []

    # Generate random shares for n-1 parties
    for _ in range(len(parties) - 1):
        r = torch.randint(low=0, high=100, size=value.shape)
        random_shares.append(r)

    # Last share ensures sum equals original value
    final_share = value - sum(random_shares)
    random_shares.append(final_share)

    # Assign shares to parties
    for party, share in zip(parties, random_shares):
        # In PySyft: share.send(party)
        # Here: simulate sending
        ptr = send_share(share, party)
        shares.append(ptr)

    return shares

# Create three simulated parties
alice = VirtualParty(name="alice")
bob = VirtualParty(name="bob")
charlie = VirtualParty(name="charlie")

parties = [alice, bob, charlie]
print("Parties created:", parties)

# 2. Data Preparation
# Each party holds a piece of the data
# We use a simple list of integers for demonstration

# Private values (known only to respective parties)
alice_value = torch.tensor([10, 20, 30])
bob_value = torch.tensor([5, 10, 15])
charlie_value = torch.tensor([2, 4, 6])

print("Private values alice_value:", alice_value)
print("Private values bob_value:", bob_value)
print("Private values charlie_value:", charlie_value)

# Secret Sharing
alice_shares = secret_share(alice_value, parties)
print("Alice shares to parties")

bob_shares = secret_share(bob_value, parties)
print("Bob shares to parties")

charlie_shares = secret_share(charlie_value, parties)
print("Charlie shares to parties")

# Local Sum at Each Party
party_sums = []

for i, party in enumerate(parties):
    local = (
        alice_shares[i] +
        bob_shares[i] +
        charlie_shares[i]
    )
    party_sums.append(local)

print("Party sums calculated")

# Reconstruct the Secure Sum
# Bring results back to the coordinator
final_sum = sum(ps.get() for ps in party_sums)

print("🔐 Secure Sum Result:", final_sum.item() if final_sum.numel() == 1 else final_sum.tolist())
