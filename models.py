import numpy as np
import random

### Helper functions ###

def kld(p, q):
  p[p == 0] = 1e-10
  q[q == 0] = 1e-10

  return sum(p * np.log(p / q))

def jsd(p, q):
  m = 0.5 * (p + q)

  return 0.5 * (kld(p, m) + kld(q, m))

def entropy(p):
  return np.where(p == 0, 0, p * -np.log(p))

def conditional_entropy(matrix):
    prob = matrix.sum(axis=0) / np.sum(matrix) #P(S)
    cond_prob = np.where(matrix.sum(axis=0) == 0, 0, matrix / matrix.sum(axis=0)) # P(r(=1:4) | S); Empty signals are equally likely to refer to any object

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
        else:
          counter += 1
          if(print_output): print(f"Cost: {old[0]}, H(R|S): {old[1]}, H(S): {old[2]}")

    return(np.round(self.mat))
  

### EXTENSION - evolve joint speaker distribution w.r.t. shared referent dimension (R, S1, S2) ###

class JointSpeakerAlignment:
  def __init__(self, size, density):
    self.size = size
    self.mat = np.zeros((self.size, self.size, self.size))
    for ref in range(self.mat.shape[0]):
      for s1 in range(self.mat.shape[1]):
        for s2 in range(self.mat.shape[2]):
          self.mat[ref,s1, s2] = random.choices([0,1], weights=(1-density, density), k=1)[0]

  def energy_function(self, matrix, lam):
    # Calculate joint entropy between speaker distributions
    joint_prob = matrix.sum(axis=0) / np.sum(matrix) # P(S1, S2)
    joint_entropy = np.sum(entropy(joint_prob)) / np.log(np.size(joint_prob)) # Hnorm(S1, S2)

    # Calculate conditional entropy of references over joint speaker distribution
    ref_cond_joint_entropy = np.sum(conditional_entropy(matrix))

    # Calculate cost
    cost = (lam * ref_cond_joint_entropy) + ((1-lam) * joint_entropy)

    return cost, ref_cond_joint_entropy, joint_entropy

  def run_to_equilibrium(self, prob, lam, stop, print_output=False):
    counter = 0
    while(counter < stop):
      trans_mat = np.zeros((self.size, self.size, self.size))
      for ref in range(trans_mat.shape[0]):
        for s1 in range(trans_mat.shape[1]):
          for s2 in range(trans_mat.shape[2]):
            trans_mat[ref, s1, s2] = random.choices([0,1], weights=(1-prob, prob), k=1)[0]

      new_mat = abs(self.mat - trans_mat)

      old = self.energy_function(self.mat, lam)
      new = self.energy_function(new_mat, lam)

      if(new[0] < old[0]):
        self.mat = new_mat
        counter = 0
        if(print_output): print(f"Cost: {new[0]}, Hnorm(R | S1, S2): {new[1]}, Hnorm(S1, S2): {new[2]}")
      else:
        counter += 1
        if(print_output): print(f"Cost: {old[0]}, Hnorm(R | S1, S2): {old[1]}, Hnorm(S1, S2): {old[2]}")

      # Calculate JSD
      # s1 = np.mean(np.mean(self.mat, axis=0), axis=0)
      # s2 = np.mean(np.mean(self.mat, axis=0), axis=1)
      # print(f"JSD: {jsd(s1, s2)}")

    return(np.round(self.mat))