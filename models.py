import numpy as np
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

  return (sig_entropy - cond_entropy)  #/ sig_entropy


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

    # Method 1: Joint speaker entropy
    joint_prob = matrix.sum(axis=0) / np.sum(matrix) # P(S1, S2)
    joint_entropy = np.sum(entropy(joint_prob)) / np.log(np.size(joint_prob)) # Hnorm(S1, S2)

    # Method 2: Avg S1 and S2 entropy
    # s1 = matrix.mean(axis=1)
    # s1_prob = s1.sum(axis=0) / np.sum(s1) # P(S)
    # s1_entropy = np.sum(entropy(s1_prob)) / np.log(np.size(s1_prob)) # Hnorm(S1)

    # s2 = matrix.mean(axis=2)
    # s2_prob = s2.sum(axis=0) / np.sum(s2) # P(S)
    # s2_entropy = np.sum(entropy(s2_prob)) / np.log(np.size(s2_prob)) # Hnorm(S2)

    # joint_entropy = np.mean([s1_entropy, s2_entropy])

    # Calculate conditional entropy of references over joint speaker distribution
    ref_cond_joint_entropy = np.sum(conditional_entropy(matrix))

    # Calculate cost
    cost = (lam * ref_cond_joint_entropy) + ((1-lam) * joint_entropy)

    return cost, ref_cond_joint_entropy, joint_entropy

  def run_to_equilibrium(self, prob, lam, stop, print_output=False):
    cost_over_time = []
    cond_entropy_over_time = []
    signal_entropy_over_time = []
    jsd_over_time = []

    counter = 0
    while(counter < stop):
      trans_mat = np.zeros((self.referent, self.signal, self.signal))
      for ref in range(trans_mat.shape[0]):
        for s1 in range(trans_mat.shape[1]):
          for s2 in range(trans_mat.shape[2]):
            trans_mat[ref, s1, s2] = random.choices([0,1], weights=(1-prob, prob), k=1)[0]

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
        s1 = self.mat.mean(axis=1)
        s2 = self.mat.mean(axis=2)

        jsd_values = []
        for i in range(self.mat.shape[0]):
          s1_prob = s1[i] / np.sum(s1[i])
          s2_prob = s2[i] / np.sum(s2[i])

          value = jsd(s1_prob, s2_prob)
          jsd_values.append(value)
        avg_jsd = np.mean(jsd_values)

        if(print_output): print(f"JSD: {avg_jsd}")
        jsd_over_time.append(avg_jsd)

    return(self.mat, cost_over_time, cond_entropy_over_time, signal_entropy_over_time, jsd_over_time)