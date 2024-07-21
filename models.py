import numpy as np
import pandas as pd
import random

### Helper functions ###

def kld(p, q):
  # q[q == 0] = 1e-10

  return np.sum(np.where(p == 0, 0, p * np.log(p / q)))

def jsd(p, q):
  m = 0.5 * (p + q)

  return 0.5 * (kld(p, m) + kld(q, m))

def entropy(p):
  return np.where(p == 0, 0, p * -np.log(p))

def conditional_entropy(matrix):
    prob = matrix.sum(axis=0) / np.sum(matrix) #P(S)
    cond_prob = np.where(matrix.sum(axis=0) == 0, 0, matrix / matrix.sum(axis=0)) # P(r(=1:4) | S)

    cond_entropy = entropy(cond_prob) / np.log(matrix.shape[0]) # Hnorm(r | S)

    return cond_entropy.sum(axis=0) * prob # Hnorm(R | S)

def mutual_information(matrix):
  sig_prob = matrix.sum(axis=0) / np.sum(matrix) # P(S)

  sig_entropy = np.sum(entropy(sig_prob)) / np.log(np.size(sig_prob))
  cond_entropy = np.sum(conditional_entropy(matrix.T))

  return (sig_entropy - cond_entropy)


### REPLICATION - Ferrer i Cancho & Sole (2002) ###

class SignalEvolution:
  def __init__(self, size, density):
    self.size = size
    self.mat = np.zeros((self.size, self.size))
    for row in range(self.mat.shape[0]):
      for col in range(self.mat.shape[1]):
        self.mat[row, col] = random.choices([0,1], weights=(1-density, density), k=1)[0]

  def energy_function(self, matrix, lam):
    # Calculate entropy of speaker distribution
    sig_prob = matrix.sum(axis=0) / np.sum(matrix) # P(S)
    sig_entropy = np.sum(entropy(sig_prob)) / np.log(np.size(sig_prob)) # Hnorm(S)

    # Calculate conditional entropy of references over speaker distribution
    ref_cond_sig_entropy = np.sum(conditional_entropy(matrix))

    # Calculate cost
    cost = (lam * ref_cond_sig_entropy) + ((1-lam) * sig_entropy)

    return cost, ref_cond_sig_entropy, sig_entropy

  def run_to_equilibrium(self, prob, lam, stop, print_output=False):
    cost_over_time = []
    cond_entropy_over_time = []
    signal_entropy_over_time = []

    counter = 0
    while(counter < stop):
      trans_mat = np.zeros((self.size, self.size))
      for row in range(trans_mat.shape[0]):
        for col in range(trans_mat.shape[1]):
          trans_mat[row, col] = random.choices([0,1], weights=(1-prob, prob), k=1)[0] 

      new_mat = abs(self.mat - trans_mat)

      if 0 not in new_mat.mean(axis=1): # Disallow signless referents
        old = self.energy_function(self.mat, lam)
        new = self.energy_function(new_mat, lam)

        if(new[0] < old[0]):
          self.mat = new_mat
          counter = 0
          if(print_output): print(f"Cost: {new[0]}, H(R|S): {new[1]}, H(S): {new[2]}")
          cost_over_time.append(new[0])
          cond_entropy_over_time.append(new[1])
          signal_entropy_over_time.append(new[2])
        else:
          counter += 1
          if(print_output): print(f"Cost: {old[0]}, H(R|S): {old[1]}, H(S): {old[2]}")
          cost_over_time.append(old[0])
          cond_entropy_over_time.append(old[1])
          signal_entropy_over_time.append(old[2])

    return(self.mat, cost_over_time, cond_entropy_over_time, signal_entropy_over_time)
  

### EXTENSION - evolve joint speaker distribution w.r.t. shared referent dimension (R, S1, S2) ###

class JointSpeakerAlignment:
  def __init__(self, signal, referent, density):
    self.signal = signal
    self.referent = referent
    self.mat = np.zeros((self.referent, self.signal, self.signal))
    for ref in range(self.mat.shape[0]):
      for s1 in range(self.mat.shape[1]):
        for s2 in range(self.mat.shape[2]):
          self.mat[ref, s1, s2] = random.choices([0,1], weights=(1-density, density), k=1)[0]

  def energy_function(self, matrix, lam):
    # Calculate joint entropy between speaker distributions

    # Method 1: Joint speaker cost function
    ## Joint entropy
    joint_prob = matrix.sum(axis=0) / np.sum(matrix) # P(S1, S2)
    joint_entropy = np.sum(entropy(joint_prob)) / np.log(np.size(joint_prob)) # Hnorm(S1, S2)

    ## Calculate conditional entropy of references over joint speaker distribution
    ref_cond_joint_entropy = np.sum(conditional_entropy(matrix))

    # # Method 2: Individualized cost function
    # ## Avg signal entropy
    # s1 = matrix.mean(axis=1)
    # s1_prob = s1.sum(axis=0) / np.sum(s1) # P(S)
    # s1_entropy = np.sum(entropy(s1_prob)) / np.log(np.size(s1_prob)) # Hnorm(S1)
 # 
    # s2 = matrix.mean(axis=2)
    # s2_prob = s2.sum(axis=0) / np.sum(s2) # P(S)
    # s2_entropy = np.sum(entropy(s2_prob)) / np.log(np.size(s2_prob)) # Hnorm(S2)
# 
    # joint_entropy = np.mean([s1_entropy, s2_entropy])
# 
    # ## Avg conditional entropy
    # rs1 = matrix.mean(axis=1)
    # rs2 = matrix.mean(axis=2)
# 
    # rs1_cond_entropy = np.sum(conditional_entropy(rs1))
    # rs2_cond_entropy = np.sum(conditional_entropy(rs2))
# 
    # ref_cond_joint_entropy = np.mean([rs1_cond_entropy, rs2_cond_entropy])

    # Calculate cost
    cost = (lam * ref_cond_joint_entropy) + ((1-lam) * joint_entropy)

    return cost, ref_cond_joint_entropy, joint_entropy

  def run_to_equilibrium(self, prob, lam, stop, print_output=False):
    cost_over_time = []
    cond_entropy_over_time = []
    signal_entropy_over_time = []
    jsd_over_time = []
    combined_entropy_over_time = []
    sparsity_over_time = []

    flips = []

    counter = 0
    while(counter < stop):
      trans_mat = np.zeros((self.referent, self.signal, self.signal))
      for ref in range(trans_mat.shape[0]):
        for s1 in range(trans_mat.shape[1]):
          for s2 in range(trans_mat.shape[2]):
            trans_mat[ref, s1, s2] = random.choices([0,1], weights=(1-prob, prob), k=1)[0]

      flips.append(np.sum(trans_mat))

      new_mat = abs(self.mat - trans_mat)

      if 0 not in new_mat.mean(axis=1).mean(axis=1): # Disallow signless referents
        old = self.energy_function(self.mat, lam)
        new = self.energy_function(new_mat, lam)

        if(new[0] < old[0]):
          self.mat = new_mat
          counter = 0
          if(print_output): print(f"Cost: {new[0]}, Hnorm(R | S1, S2): {new[1]}, Hnorm(S1, S2): {new[2]}")
          cost_over_time.append(new[0])
          cond_entropy_over_time.append(new[1])
          signal_entropy_over_time.append(new[2])
        else:
          counter += 1
          if(print_output): print(f"Cost: {old[0]}, Hnorm(R | S1, S2): {old[1]}, Hnorm(S1, S2): {old[2]}")
          cost_over_time.append(old[0])
          cond_entropy_over_time.append(old[1])
          signal_entropy_over_time.append(old[2])

        # Calculate JSD
        s1 = self.mat.mean(axis=2)
        s2 = self.mat.mean(axis=1)

        s1_prob = s1.sum(axis=0) / np.sum(s1.sum(axis=0))
        s2_prob = s2.sum(axis=0) / np.sum(s2.sum(axis=0))

        value = jsd(s1_prob, s2_prob)
        jsd_over_time.append(value)

        # Calculate H(S1+S2)
        s1s2_prob = (np.array(s1_prob) + np.array(s2_prob)) / 2
        combined_entropy = np.sum(entropy(s1s2_prob)) / np.log(np.size(s1s2_prob))
        combined_entropy_over_time.append(combined_entropy)

        # Track matrix sparsity over time
        sparsity_over_time.append(self.mat.sum() / np.prod(self.mat.shape))
        print(self.mat.sum())

    print(np.mean(flips))

    return(self.mat, cost_over_time, cond_entropy_over_time, signal_entropy_over_time, jsd_over_time, combined_entropy_over_time, sparsity_over_time)
  

### 4D - evolve joint speaker distribution w.r.t. *individual* referent dimensions (R1, R2, S1, S2) ###

class ReferentialAlignment:
  def __init__(self, signal, referent, density):
    self.signal = signal
    self.referent = referent
    self.mat = np.zeros((self.referent, self.referent, self.signal, self.signal))
    for r1 in range(self.mat.shape[0]):
      for r2 in range(self.mat.shape[1]):
        for s1 in range(self.mat.shape[2]):
          for s2 in range(self.mat.shape[3]):
            self.mat[r1, r2, s1, s2] = random.choices([0,1], weights=(1-density, density), k=1)[0]

  def energy_function(self, matrix, lam):
    # Method 1: Joint speaker cost function
    ## Calculate joint entropy between speaker distributions
    joint_prob = matrix.sum(axis=0) / np.sum(matrix) # P(S1, S2)
    joint_entropy = np.sum(entropy(joint_prob)) / np.log(np.size(joint_prob)) # Hnorm(S1, S2)

    ## Calculate conditional entropy of references over joint speaker distribution
    ref_cond_joint_entropy = np.sum(conditional_entropy(matrix))

    # # Method 2: Individualized cost function
    # ## Avg signal entropy
    # s1 = matrix.mean(axis=1)
    # s1_prob = s1.sum(axis=0) / np.sum(s1) # P(S)
    # s1_entropy = np.sum(entropy(s1_prob)) / np.log(np.size(s1_prob)) # Hnorm(S1)
 # 
    # s2 = matrix.mean(axis=2)
    # s2_prob = s2.sum(axis=0) / np.sum(s2) # P(S)
    # s2_entropy = np.sum(entropy(s2_prob)) / np.log(np.size(s2_prob)) # Hnorm(S2)
    # 
    # joint_entropy = np.mean([s1_entropy, s2_entropy])
# 
    # ## Avg conditional entropy
    # rs1 = matrix.mean(axis=1)
    # rs2 = matrix.mean(axis=2)
# 
    # rs1_cond_entropy = np.sum(conditional_entropy(rs1))
    # rs2_cond_entropy = np.sum(conditional_entropy(rs2))
# 
    # ref_cond_joint_entropy = np.mean([rs1_cond_entropy, rs2_cond_entropy])

    # Calculate cost
    cost = (lam * ref_cond_joint_entropy) + ((1-lam) * joint_entropy)

    return cost, ref_cond_joint_entropy, joint_entropy

  def run_to_equilibrium(self, prob, lam, stop):
    cost_over_time = []
    cond_entropy_over_time = []
    signal_entropy_over_time = []
    jsd_over_time = []
    ref_align_mi = []
    combined_entropy_over_time = []
    sparsity_over_time = []
    mat_over_time = [self.mat.flatten()]

    flips = []

    timestep = 0
    counter = 0
    discarded = 0
    while(counter < stop):
      # Save current matrix
      timestep += 1
      if (timestep % 100): 
        mat_over_time.append(self.mat.flatten())
        # print(self.mat)

      # Generate competitor
      trans_mat = np.zeros((self.referent, self.referent, self.signal, self.signal))
      for r1 in range(trans_mat.shape[0]):
        for r2 in range(trans_mat.shape[1]):
          for s1 in range(trans_mat.shape[2]):
            for s2 in range(trans_mat.shape[3]):
              trans_mat[r1, r2, s1, s2] = random.choices([0,1], weights=(1-prob, prob), k=1)[0]

      flips.append(np.sum(trans_mat))

      new_mat = abs(self.mat - trans_mat)

      if ((0 not in new_mat.mean(axis=0).mean(axis=1).mean(axis=1)) and (0 not in new_mat.mean(axis=1).mean(axis=1).mean(axis=1))): # Disallow signless referents
        print(f'Discarded Matrices: {discarded}')
        discarded = 0 

        old_s1 = self.energy_function(self.mat.mean(axis=1), lam) # Avg over S2 referent dimension
        old_s2 = self.energy_function(self.mat.mean(axis=0), lam) # mutatis mutandis...
        new_s1 = self.energy_function(new_mat.mean(axis=1), lam)
        new_s2 = self.energy_function(new_mat.mean(axis=0), lam)

        old_cost = np.mean([old_s1[0], old_s2[0]])
        new_cost = np.mean([new_s1[0], new_s2[0]])

        if(new_cost < old_cost):
          self.mat = new_mat
          counter = 0
          cost_over_time.append([new_s1[0], new_s2[0]])
          cond_entropy_over_time.append([new_s1[1], new_s2[1]])
          signal_entropy_over_time.append([new_s1[2], new_s2[2]])
          print(f'Cost: {new_cost}; Conditional Entropy: {np.mean([new_s1[1], new_s2[1]])}; Signal Entropy: {new_s1[2]}')

        else:
          counter += 1
          cost_over_time.append([old_s1[0], old_s2[0]])
          cond_entropy_over_time.append([old_s1[1], old_s2[1]])
          signal_entropy_over_time.append([old_s1[2], old_s2[2]])
          print(f'Cost: {old_cost}; Conditional Entropy: {np.mean([old_s1[1], old_s2[1]])}; Signal Entropy: {old_s1[2]}')

        # Calculate JSD (between speakers)
        s1 = self.mat.mean(axis=1).mean(axis=2)
        s2 = self.mat.mean(axis=0).mean(axis=1)

        s1_prob = s1.sum(axis=0) / np.sum(s1.sum(axis=0))
        s2_prob = s2.sum(axis=0) / np.sum(s2.sum(axis=0))

        value = jsd(s1_prob, s2_prob)
        jsd_over_time.append(value)

        # Measure alignment of referent dimensions
        ref_matrix = self.mat.mean(axis=3).mean(axis=2)
        ref_align_mi.append([mutual_information(ref_matrix)])

        # Calculate H(S1+S2)
        s1s2_prob = (np.array(s1_prob) + np.array(s2_prob)) / 2
        combined_entropy = np.sum(entropy(s1s2_prob)) / np.log(np.size(s1s2_prob))
        combined_entropy_over_time.append(combined_entropy)

        # Track matrix sparsity over time
        sparsity_over_time.append(self.mat.sum() / np.prod(self.mat.shape))
        print(f'Sparsity: {self.mat.sum()}')

      else:
        discarded += 1

    print(np.mean(flips))
    return(self.mat, cost_over_time, cond_entropy_over_time, signal_entropy_over_time, jsd_over_time, ref_align_mi, combined_entropy_over_time, sparsity_over_time, mat_over_time)
