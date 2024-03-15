import numpy as np
import pandas as pd

class MixtureModel:
    def __init__(self, p, q, num_trials, num_participants, pix, piy, c, A, account_confusion, account_trust):
        self.p = p
        self.q = q
        self.num_trials = num_trials
        self.num_participants = num_participants
        self.pix = pix
        self.piy = piy
        self.c = c
        self.A = A
        self.account_confusion = account_confusion
        self.account_trust = account_trust

    def run_trial(self):
        probs = np.array([1 - self.pix, self.pix, self.piy])
        weights = np.array([self.q, self.p, 1 - self.p - self.q])
        question_num = np.random.choice([0, 1, 2], size=self.num_participants, p=weights)
        is_yes = flip_answer(np.zeros(self.num_participants, dtype=bool), probs[question_num])
        apply_trust = np.logical_or((np.logical_and(question_num == 0, np.logical_not(is_yes))), np.logical_and(question_num == 1, is_yes > 0))
        is_yes[apply_trust] = flip_answer(is_yes[apply_trust], 1 - self.A)
        is_yes[:] = flip_answer(is_yes, self.c)
        num_yes = np.sum(is_yes)
        Py_hat = num_yes / self.num_participants
        return Py_hat
    
    def compute_theoretical_values(self):
        rigged_c = self.rigged_confusion_question()
        rigged_A = self.rigged_trust_question()

        def compute_Py(model):
            if model.account_confusion:
                q_term = model.q * (1 - model.c + model.A*(-1 + 2*model.c)*model.pix)
                p_term = model.p * (model.c + model.A*model.pix - 2*model.A*model.c*model.pix)
                qp1_term = (1 - model.p - model.q) * (model.c + model.piy - 2*model.c*model.piy) 
                return q_term + p_term + qp1_term
            else:
                q_term = model.q * (1 - model.A*model.pix)
                p_term = model.p * (model.A*model.pix)
                qp1_term = (1 - model.p - model.q) * (model.piy) 
                return q_term + p_term + qp1_term

        Py = compute_Py(self)
        Pyc = compute_Py(rigged_c) 
        PyA = compute_Py(rigged_A)
        var_Py_hat = (Py * (1 - Py)) / self.num_participants
        eta1 = (self.pix*(self.piy - self.p*self.piy + self.A*(self.p - self.q) + self.q - self.piy*self.q + 
                          self.c*(1 - 2*self.A*(self.p - self.q) - 2*self.q + 2*self.piy*(-1 + self.p + self.q)))) / Py
        eta2 = (self.pix*(1 - self.piy + self.p*self.piy - self.q + self.piy*self.q + self.A*(-self.p + self.q) +
                          self.c*(-1 + 2*self.A*(self.p-self.q) + 2*self.q - 2*self.piy*(-1 + self.p + self.q))) ) / (1 - Py)
        pp = (1 - max(eta1, eta2)) / (1 - self.pix)

        if self.account_confusion and self.account_trust:
            var_c_hat = (Pyc * (1-Pyc)) / (rigged_c.num_participants * (1 - 2*rigged_c.q + 2*rigged_c.piy*(-1 + rigged_c.p + rigged_c.q))**2)
            var_A_hat = (PyA * (1-PyA)) / (rigged_A.num_participants * rigged_A.p**2) 
            Py_term = var_Py_hat / (self.A * (1 - 2*self.c) * (self.p - self.q))**2
            c_term = var_c_hat * ((2*Py - 1) / (self.A * (self.p - self.q) * (1 - 2*self.c)**2))**2
            A_term = var_A_hat * ((Py - self.c * (1 - 2*self.q) - self.q - self.piy * (1 - self.p - self.q) * (1 - 2*self.c)) /
                                  (self.A**2 * (1 - 2*self.c) * (self.p - self.q)))**2

            var_pix_hat = Py_term + c_term + A_term
        elif self.account_trust:
            var_c_hat = 0
            var_A_hat = (PyA * (1-PyA)) / (rigged_A.num_participants * rigged_A.p**2) 
            Py_term = var_Py_hat * (1 / (self.A * (self.p - self.q)))**2
            A_term = var_A_hat * ((-Py + self.q - self.piy*(-1 + self.p + self.q)) / (self.A**2 * (self.p - self.q)))**2
            var_pix_hat = Py_term + A_term
            bias = -2*self.c*self.pix + ((self.c * (1 - 2*self.q + 2*self.piy*(-1 + self.p + self.q))) / (self.A * (self.p - self.q)))
            var_pix_hat += bias**2
        elif self.account_confusion:
            assert False, 'We will not be using this feature.'
        else:
            assert False, 'We will not be using this feature.'

        a, b = 1, 1
        unified = pp**a / var_pix_hat**b
        return var_c_hat, var_A_hat, var_pix_hat, pp, unified
    
    def rigged_confusion_question(self):
        return MixtureModel(self.p, self.q, self.num_trials, self.num_participants, 0, self.piy, self.c, 1, self.account_confusion, self.account_trust)
    
    def rigged_trust_question(self):
        return GreenbergModel(0.7, self.num_trials, self.num_participants, 1, self.piy, 0, self.A,
                            self.account_confusion, self.account_trust)

    def compute_c_hat(self, Py_hat, _, __):
        return (Py_hat - self.q - self.piy * (1 - self.p - self.q)) / (1 - 2*self.q - 2*self.piy*(1 - self.p - self.q))
    
    def compute_A_hat(self, Py_hat, _, __):
        return (Py_hat - (1-self.p)*self.piy) / self.p

    def compute_pi_hat(self, Py_hat, c_hat, A_hat):
        return (-Py_hat + c_hat*(1 - 2*self.q) + self.piy*(1 - 2*c_hat)*(1 - self.p - self.q) + self.q) / (A_hat*(-1 + 2*c_hat)*(self.p - self.q))
    
    def actual_pi(self):
        return self.pix
    
    def get_pqcApi(self):
        return [self.p, self.q, self.c, self.A, self.pix]
    

class WarnerModel(MixtureModel):
    def __init__(self, p, num_trials, num_participants, pi, c, A, account_confusion, account_trust):
        super().__init__(p, 1-p, num_trials, num_participants, pi, 0, c, A, account_confusion, account_trust)
    

class GreenbergModel(MixtureModel):
    def __init__(self, p, num_trials, num_participants, pix, piy, c, A, account_confusion, account_trust):
        super().__init__(p, 0, num_trials, num_participants, pix, piy, c, A, account_confusion, account_trust)


class SimulationOutput:
    def __init__(self, model, c_hat, var_c_hat, var_hat_c_hat, A_hat, var_A_hat, var_hat_A_hat, pi_hat, var_pi_hat, var_hat_pi_hat, pp, unified):
        self.model = model
        self.c_hat = c_hat
        self.var_c_hat = var_c_hat
        self.var_hat_c_hat = var_hat_c_hat
        self.A_hat = A_hat
        self.var_A_hat = var_A_hat
        self.var_hat_A_hat = var_hat_A_hat
        self.pi_hat = pi_hat
        self.var_pi_hat = var_pi_hat
        self.var_hat_pi_hat = var_hat_pi_hat
        self.pp = pp
        self.unified = unified


def generate_table(list_of_models):
    headers = ['$p$', '$q$', '$m$', '$m_hat$', '$\MSE[[\hat{m}]$', '$\widehat{\MSE[[\hat{m}]}$', '$A$', '$\hat{A}$', '$\MSE[[\hat{A}]$', 
               '$\widehat{\MSE[[\hat{A}]}$', '$\pi_x$', '$\hat{\pi}_x$', '$\MSE[[\hat{\pi}_x]$', '$\widehat{\MSE[[\hat{pi}_x]}$',
               '$PP$', '$\mathbb{M}$']
    results = np.empty((len(list_of_models), len(headers)))

    for i, model in enumerate(list_of_models):
        output = run_simulation(model)
        formatted_output = np.array(output.model.get_pqcApi() + list(vars(output).values())[1:])
        results[i] = formatted_output[np.array([0, 1, 2, 5, 6, 7, 3, 8, 9, 10, 4, 11, 12, 13, 14, 15])]

    table = pd.DataFrame(results, columns=headers)
    return table


def run_simulation(model):
    rigged_confusion_question = model.rigged_confusion_question()
    rigged_trust_question = model.rigged_trust_question()

    if model.account_confusion and model.account_trust:
        c_hat, var_hat_c_hat = ask_question(rigged_confusion_question, rigged_confusion_question.compute_c_hat, None, None)
        A_hat, var_hat_A_hat = ask_question(rigged_trust_question, rigged_trust_question.compute_A_hat, None, None)
    elif model.account_confusion:
        c_hat, var_hat_c_hat = ask_question(rigged_confusion_question, rigged_confusion_question.compute_c_hat, None, None)
        A_hat, var_hat_A_hat = 1, 0
    elif model.account_trust:
        c_hat = var_hat_c_hat = 0
        A_hat, var_hat_A_hat = ask_question(rigged_trust_question, rigged_trust_question.compute_A_hat, None, None)
    else:
        c_hat = var_hat_c_hat = 0
        A_hat, var_hat_A_hat = 1, 0
    pi_hat, var_hat_pi_hat = ask_question(model, model.compute_pi_hat, c_hat, A_hat)
    var_c_hat, var_A_hat, var_pi_hat, pp, unified = model.compute_theoretical_values()

    return SimulationOutput(model, c_hat, var_c_hat, var_hat_c_hat, A_hat, var_A_hat, var_hat_A_hat, pi_hat, var_pi_hat, var_hat_pi_hat, pp, unified)


def ask_question(model, compute_statistic, c_hat, A_hat):
    Py_hat_list = np.array([model.run_trial() for _ in range(model.num_trials)])
    stat_hat_list = compute_statistic(Py_hat_list, c_hat, A_hat)
    stat_hat = np.mean(stat_hat_list)
    if (model.account_confusion and model.account_trust) or (not model.account_confusion and A_hat is None):
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