import numpy as np
import pandas as pd

class MixtureModel:
    def __init__(self, p, q, num_trials, num_participants, pix, piy, c, account_confusion):
        self.p = p
        self.q = q
        self.num_trials = num_trials
        self.num_participants = num_participants
        self.pix = pix
        self.piy = piy
        self.c = c
        self.account_confusion = account_confusion

    def run_trial(self):
        probs = np.array([1 - self.pix, self.pix, self.piy])
        weights = np.array([self.q, self.p, 1 - self.p - self.q])
        question_num = np.random.choice(np.arange(0, 3), size=self.num_participants, p=weights)
        is_yes = flip_answer(np.zeros(self.num_participants, dtype=bool), probs[question_num])
        is_yes[question_num <= 1] = flip_answer(is_yes[question_num <= 1], self.c)
        num_yes = np.sum(is_yes)
        Py_hat = num_yes / self.num_participants
        
        return Py_hat
    
    def compute_theoretical_values(self):
        rigged = self.rigged_question()
        Py = (self.p * (self.pix - self.piy) + self.piy - self.c * (-1 + 2 * self.pix) * (self.p - self.q)
              + self.q - self.q * (self.pix + self.piy))
        Py0 = (rigged.p - rigged.q)*(1 - rigged.c) + rigged.q + rigged.piy*(1 - rigged.p - rigged.q)
        var_Py_hat = (Py * (1 - Py)) / self.num_participants
        if self.account_confusion:
            var_c_hat = (Py0 * (1 - Py0)) / (self.num_participants * (rigged.p - rigged.q)**2)
            var_pix_hat = (var_Py_hat / ((1 - 2 * self.c) * (self.p - self.q))**2 +
                            (var_c_hat) * ((self.p - self.q - 2*(Py - self.q - (1 - self.p - self.q)*self.piy)) /
                                        ((1 - 2*self.c)**2 * (self.p - self.q))) ** 2)
        else:
            var_c_hat = 0
            var_pix_hat = (Py * (1-Py)) / ((self.num_participants-1) * (self.p-self.q)**2)
            bias_pix_hat = self.c * (1 - 2*self.pix)
            var_pix_hat += bias_pix_hat**2

        return var_c_hat, var_pix_hat
    
    def rigged_question(self):
        return MixtureModel(self.p, self.q, self.num_trials, self.num_participants, 1, 0, self.c, self.account_confusion)
    
    def compute_c_hat(self, Py_hat, _):
        return (self.p - Py_hat + self.piy*(1 - self.p - self.q)) / (self.p - self.q)
    
    def compute_pi_hat(self, Py_hat, c_hat):
        return ((self.piy - Py_hat + c_hat * (self.p - self.q) + self.q - self.piy * (self.p + self.q)) /
                ((2 * c_hat - 1) * (self.p - self.q)))
    
    def actual_pi(self):
        return self.pix
    
    def get_pqc(self):
        return [self.p, self.q, self.c]
    
class WarnerModel(MixtureModel):
    def __init__(self, p, num_trials, num_participants, pi, c, account_confusion):
        super().__init__(p, 1-p, num_trials, num_participants, pi, 0, c, account_confusion)
    
class GreenbergModel(MixtureModel):
    def __init__(self, p, num_trials, num_participants, pix, piy, c, account_confusion):
        super().__init__(p, 0, num_trials, num_participants, pix, piy, c, account_confusion)

class SimulationOutput:
    def __init__(self, model, c_hat, var_c_hat, var_hat_c_hat, pi_hat, var_pi_hat, var_hat_pi_hat):
        self.model = model
        self.c_hat = c_hat
        self.var_c_hat = var_c_hat
        self.var_hat_c_hat = var_hat_c_hat
        self.pi_hat = pi_hat
        self.var_pi_hat = var_pi_hat
        self.var_hat_pi_hat = var_hat_pi_hat

def generate_table(list_of_models):
    headers = ['p', 'q', 'c', 'c_hat',
               'var_c_hat', 'var_hat_c_hat', 'pi_hat', 'var_pi_hat', 'var_hat_pi_hat']
    results = np.empty((len(list_of_models), len(headers)))

    for i, model in enumerate(list_of_models):
        output = run_simulation(model)
        results[i] = output.model.get_pqc() + list(vars(output).values())[1:]

    table = pd.DataFrame(results, columns=headers)
    
    return table

def run_simulation(model):
    rigged_question = model.rigged_question()
    if model.account_confusion:
        c_hat, var_hat_c_hat = ask_question(rigged_question, rigged_question.compute_c_hat, None)
    else:
        c_hat = var_hat_c_hat = 0
    pi_hat, var_hat_pi_hat = ask_question(model, model.compute_pi_hat, c_hat)
    var_c_hat, var_pi_hat = model.compute_theoretical_values()

    return SimulationOutput(model, c_hat, var_c_hat, var_hat_c_hat, pi_hat, var_pi_hat, var_hat_pi_hat)

def ask_question(model, compute_statistic, c_hat):
    Py_hat_list = np.array([model.run_trial() for _ in range(model.num_trials)])
    stat_hat_list = compute_statistic(Py_hat_list, c_hat)
    stat_hat = np.mean(stat_hat_list)
    if model.account_confusion:
        var_hat_stat_hat = np.var(stat_hat_list, ddof=1)
    else:
        var_hat_stat_hat = mse(np.full(model.num_trials, model.actual_pi()), stat_hat_list)
    
    return stat_hat, var_hat_stat_hat

def flip_answer(bool_vals, prob_flip):
    rand_nums = np.random.random(size=bool_vals.shape)
    bool_vals[(0 <= rand_nums) & (rand_nums < prob_flip)] = ~bool_vals[(0 <= rand_nums) & (rand_nums < prob_flip)]
    return bool_vals

def mse(actual, predicted):
    return np.square(np.subtract(np.array(actual), np.array(predicted))).mean()