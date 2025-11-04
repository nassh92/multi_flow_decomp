from abc import ABC, abstractmethod
import numpy as np
import random
import copy
import math
import sys
import os
sys.path.append(os.getcwd())
from utils.graph_utils import Node 



#########################################################################################
################################    Constants    ########################################
#########################################################################################
POLICY_BASED_TYPE_AGENTS = {"random", "LRI", "LRP"}

VALUE_BASED_TYPE_AGENTS = {"Eps_exp0", "Eps_exp1", "Eps_exp2", "UCB"}

MIXED_TYPE_AGENTS = {"Pursuit", "Hier_cont_pursuit", "Exp3"}




###################################################################################################################
############################################### Classes abstraites RL stateless ###################################
###################################################################################################################
class BanditAgent(ABC):
    """
    Classe abstraite representant un agent d'apprentissage par renforcement sans etats
    
    """
    def __init__(self, actions, policy):
        """
            actions : liste contenant l'espace d'action
            policy : liste contenant les valeurs/probabilité associées aux actions
        """
        self.actions = actions
        # 'self.actions_to_inds' is not needed inside an object of type 'BanditAgent' and is only used outside (if useful for processing)
        self.actions_to_inds = {self.actions[ind_action]:ind_action for ind_action in range(len(self.actions))}
        self.policy = copy.deepcopy(return_initial_policy(policy, len(self.actions)))
        
        if self.policy is not None and self.actions is not None:
            if np.sum(self.policy) < 0.9999 or np.sum(self.policy) > 1.0001:
                print("The policy ", policy)
                raise Exception("La somme des probas ne vaut pas 1, elle vaut "+str(np.sum(self.policy)))
            
            if len(self.policy) != len(self.actions):
                print("the actions ",self.actions)
                print("the policy ", self.policy)
                print("agent type ", type(self).__name__)
                print("length pol/actions", len(self.policy), len(self.actions))
                raise Exception("La taille de la politique et incohérent avec le nombre d'actions.")
    
    def reset_agent(self, initial_policy, initial_hyperparams):
        """
        reinitialise la politique apres une execution
        """
        self.policy = copy.deepcopy(return_initial_policy(initial_policy, len(self.actions)))
    
    @abstractmethod
    def chose_action(self):
        """
        Methode abstraite implementant le choix d'une action suivant le type d'agent
        """
        pass
    
    @abstractmethod
    def update_policy(self, id_action, reward):
        """
        Mise à jour de la politique
        """
        pass




##################################################################################################################
############################################### Algorithmes simple et stateless ##################################
##################################################################################################################
class RandomAgent(BanditAgent):
    
    def __init__(self, actions, policy):
        super().__init__(actions, policy)
        
    def chose_action(self, id_end=None):
        """
        Choisit l'action en échantillonant le vecteur de probabilité
        """
        if id_end is None:
            id_chosen_action = np.random.choice(len(self.policy), p = self.policy)
        else:
            id_chosen_action = np.random.choice(np.arange(0, id_end+1), 
                                            p = self.policy[:id_end+1]/np.sum(self.policy[:id_end+1]))
        return id_chosen_action
    
    def update_policy(self, id_action, reward):
        pass



class EpsilonGreedy(BanditAgent):
    """
    Classe représentant un agent non contextuel Epsilon Greedy.
       
    paramètres:
        learning_rate : le taux suivant lequel les stratégies sont mise à jours
    """
    
    def __init__(self, actions, epsilon, actions_estimates, exploration=0, min_epsilon = 0.0025):
        super().__init__(actions, None)
        self.exploration = exploration
        self.initial_epsilon = epsilon
        self.min_eps = min_epsilon
        if self.initial_epsilon < self.min_eps or self.initial_epsilon > 1.0:
            raise Exception("La valeur de epsilon est trop faible.")
        self.epsilon = epsilon
        # Les données sur les estimations des actions
        nb_actions = len(self.actions)
        self.actions_estimates = copy.deepcopy(actions_estimates)
        self.nb_selected_times = np.array([0.0]*nb_actions)
        
        
    def reset_agent(self, initial_policy, initial_hyperparams):
        """
        reinitialise la politique apres une execution
        """
        super().reset_agent(None, initial_hyperparams)
        self.epsilon = self.initial_epsilon
        self.actions_estimates = return_initial_actions_estimates(initial_hyperparams, "non_hier", len(self.actions))
        self.nb_selected_times = np.array([0.0]*len(self.actions))
        
        
    def chose_action(self, id_end=None):
        """
        Choisit l'action selon en échantillonant le vecteur de probabilité
        """
        if id_end==None:
            id_end=len(self.actions)-1
        
        if random.random() < self.epsilon:
            id_chosen_action = random.randint(0, id_end)
        else:
            list_index_val_max = [i for i, v in enumerate(self.actions_estimates) if v == max(self.actions_estimates[0:id_end+1]) and i <= id_end]
            id_chosen_action = list_index_val_max[random.randint(0, len(list_index_val_max)-1)]
        
        return id_chosen_action     


    def update_policy(self, id_action, reward):
        """
        Mise à jour du vecteur de probabilité
        id_action : l'indice de l'action appliquée sur l'environnement servant à la mise à jour de la politique
        reward : la reward retournée par application de l'action d'indice id_action
        """
        # estimate the rewards
        self.nb_selected_times[id_action] +=  1
        self.actions_estimates[id_action] += (1/self.nb_selected_times[id_action])*(reward - \
                                                        self.actions_estimates[id_action])
        if self.exploration == 0:
            pass
        elif self.exploration == 1:
            self.epsilon = max(min((len(self.actions)/2)*self.initial_epsilon/np.sum(self.nb_selected_times), 1.0), self.min_eps)
        elif self.exploration == 2:
            self.epsilon = max(self.epsilon/2, self.min_eps) 
        else:
            raise Exception("Exploration parameter unrecognized.")
            
            
class UCBAgent(BanditAgent):
    """
    Classe représentant un agent non contextuel UCB.
        UCB represente la politique a l'aide de la valeur des politiques + une upper confidence.
    paramètres:
        learning_rate : le taux suivant lequel les stratégies sont mise à jours
    """
    
    def __init__(self, actions, epsilon, actions_estimates):
        super().__init__(actions, None)
        self.uc = epsilon
        # Les données sur les estimations des actions
        nb_actions = len(self.actions)
        self.actions_estimates = copy.deepcopy(actions_estimates)
        self.nb_selected_times = np.array([1.0]*nb_actions)
        self.sum_nb_selected_times = np.sum(self.nb_selected_times)
        
        
    def reset_agent(self, initial_policy, initial_hyperparams):
        """
        reinitialise la politique apres une execution
        """
        super().reset_agent(None, initial_hyperparams)
        self.actions_estimates = return_initial_actions_estimates(initial_hyperparams, "non_hier", len(self.actions))
        self.nb_selected_times = np.array([1.0]*len(self.actions))
        
        
    def __ucb_value_function(self, id_action):
        """
        Calcul la valeur de l'action d'indice id_action (val + uc * sqrt( ln( sum(nb_selected_times) / n_id_action ) ) )
        """
        ac_value = self.actions_estimates[id_action]
        ucb_val = ac_value + self.uc * math.sqrt( math.log(self.sum_nb_selected_times) / self.nb_selected_times[id_action])
        return ucb_val
    
    
    def chose_action(self, id_end=None):
        """
        Choisit l'action selon en échantillonant le vecteur de probabilité
        """
        if id_end==None:
            id_end=len(self.actions)-1
        
        ucb_values = [self.__ucb_value_function(id_ac) for id_ac in range(len(self.actions))]
        list_index_val_max = [i for i, v in enumerate(ucb_values) if v == max(ucb_values[0:id_end+1]) and i <= id_end]
        try:
            id_chosen_action = list_index_val_max[random.randint(0, len(list_index_val_max)-1)]
        except ValueError:
            print("uc param ",self.uc)
            print("ucb ", ucb_values)
            print("id_end ", id_end)
            print("List ", list_index_val_max)
            sys.exit()
        
        return id_chosen_action     


    def update_policy(self, id_action, reward):
        """
        Mise à jour du vecteur de probabilité
        id_action : l'indice de l'action appliquée sur l'environnement servant à la mise à jour de la politique
        reward : la reward retournée par application de l'action d'indice id_action
        """
        # estimate the rewards
        self.nb_selected_times[id_action] +=  1
        self.sum_nb_selected_times += 1
        self.actions_estimates[id_action] += (1/self.nb_selected_times[id_action])*(reward - \
                                                        self.actions_estimates[id_action])

        
class Exp3Agent(BanditAgent):
    """
    Classe représentant un agent non contextuel Exp3.
    Exp3 represente la politique a l'aide de la valeur des politiques.
    paramètres:
        learning_rate : le taux suivant lequel les stratégies sont mise à jours
    """
    
    def __init__(self, actions, epsilon, actions_estimates):
        super().__init__(actions, None)
        self.gamma = epsilon
        self.initial_gamma = epsilon
        # Les données sur les estimations des actions
        nb_actions = len(self.actions)
        self.actions_estimates = copy.deepcopy(actions_estimates)
        # La politique depend des valeurs dans exp3
        weights = np.exp ( np.multiply( self.gamma/nb_actions, self.actions_estimates) )
        self.policy = (1 - self.gamma) * weights/np.sum(weights) + self.gamma/nb_actions
        
        
    def reset_agent(self, initial_policy, initial_hyperparams):
        """
        reinitialise la politique apres une execution
        """
        super().reset_agent(None, initial_hyperparams)
        self.actions_estimates = return_initial_actions_estimates(initial_hyperparams, "non_hier", len(self.actions))
        # La politique depend des valeurs dans exp3
        nb_actions = len(self.actions)
        weights = np.exp ( np.multiply( self.gamma/nb_actions, self.actions_estimates) )
        self.policy = (1 - self.gamma) * weights/np.sum(weights) + self.gamma/nb_actions
    
    
    def chose_action(self, id_end=None):
        """
        Choisit l'action selon en échantillonant le vecteur de probabilité
        """
        if id_end is None:
            id_chosen_action = np.random.choice(len(self.policy), p = self.policy)
        else:
            id_chosen_action = np.random.choice(np.arange(0, id_end+1), 
                                            p = self.policy[:id_end+1]/np.sum(self.policy[:id_end+1]))
        return id_chosen_action
     
        
    def update_policy(self, id_action, reward):
        """
        Mise à jour du vecteur de probabilité
        id_action : l'indice de l'action appliquée sur l'environnement servant à la mise à jour de la politique
        reward : la reward retournée par application de l'action d'indice id_action
        """
        # estimate the rewards
        self.actions_estimates[id_action] += reward/self.policy[id_action]
        nb_actions = len(self.policy)
        weights = np.exp ( np.multiply( self.gamma/nb_actions, self.actions_estimates) )
        self.policy = (1 - self.gamma) * weights/np.sum(weights) + self.gamma/nb_actions
        


class LRAgent(BanditAgent):
    """
    Classe représentant un agent non contextuel LRI.
        LRI represente la politique a l'aide dun vecteur de proba qu'il met a jour suivant la recompense retournee.
    paramètres:
        learning_rate : le taux suivant lequel les stratégies sont mise à jours
    """
    
    def __init__(self, actions, policy, learning_rate, learning_rate_r0 = None, ag_type="LRI"):
        super().__init__(actions, policy)
        self.learning_rate = learning_rate
        if ag_type=="LRP":
            self.learning_rate_r0 = learning_rate_r0
        else:
            self.learning_rate_r0 = None
        
    def chose_action(self, id_end=None):
        """
        Choisit l'action selon en échantillonant le vecteur de probabilité
        """
        if id_end is None:
            id_chosen_action = np.random.choice(len(self.policy), p = self.policy)
        else:
            id_chosen_action = np.random.choice(np.arange(0, id_end+1), 
                                            p = self.policy[:id_end+1]/np.sum(self.policy[:id_end+1]))
        return id_chosen_action
    
    def update_policy(self, id_action, reward):
        """
        Mise à jour du vecteur de probabilité
        id_action : l'indice de l'action appliquée sur l'environnement servant à la mise à jour de la politique
        reward : la reward retournée par application de l'action d'indice id_action
        """
        nb_actions = len(self.actions)
        if reward > 0:
            u_p_vec = np.zeros(nb_actions)
            u_p_vec[id_action] = 1
            self.policy += self.learning_rate * reward * (u_p_vec - self.policy)
        else:
            if self.learning_rate_r0 is not None:
                u_p_vec = np.full((nb_actions,), self.learning_rate_r0/(nb_actions-1))
                u_p_vec[id_action] = 0
                self.policy = (1-self.learning_rate_r0)*self.policy+u_p_vec
        

class PursuitAgent (BanditAgent):
    
    '''
        Classe représentant un agent non contextuel Poursuite.
        Poursuite represente la politique a l'aide dun vecteur de proba qu'il met a jour suivant la recompense retournee.
        paramètres:
            learning_rate : le taux suivant lequel les stratégies sont mise à jours
            epsilon : probabilité de choisir aléatoirement l'action a renforcer
            actions_estimates : vecteur contenant les estimations des rewards des actions
        Remarque: la valeur pardéfaut de actions estimates est le vecteur nulle (si celle-ci n'est pas passée en entrée dans l'init ou le reset.
    '''
    
    def __init__(self, actions, policy, learning_rate, epsilon, actions_estimates):
        super().__init__(actions, policy)
        # Les pramètres de learning rate et d'exploration
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        # Les données sur les estimations des actions
        nb_actions = len(self.policy)
        self.actions_estimates = copy.deepcopy(actions_estimates)
        self.nb_selected_times = np.array([0.0]*nb_actions)
      
    
    def reset_agent(self, initial_policy, initial_hyperparams):
        """
        reinitialise la politique apres une execution
        """
        super().reset_agent(initial_policy, initial_hyperparams)
        self.actions_estimates = return_initial_actions_estimates(initial_hyperparams, "non_hier", len(self.policy))
        self.nb_selected_times = np.array([0.0]*len(self.policy))
    
    
    def chose_action(self, id_end=None):
        """
        Choisit l'action selon en échantillonant le vecteur de probabilité
        """
        if id_end is None:
            id_chosen_action = np.random.choice(len(self.policy), p = self.policy)
        else:
            id_chosen_action = np.random.choice(np.arange(0, id_end+1), 
                                            p = self.policy[:id_end+1]/np.sum(self.policy[:id_end+1]))
        
        return id_chosen_action
    
    
    def update_policy(self, id_action, reward):
        """
        Mise à jour du vecteur de probabilité
        id_action : l'indice de l'action appliquée sur l'environnement servant à la mise à jour des estimations
        reward : la reward retournée par application de l'action d'indice id_action
        """
        # estimate the rewards
        self.nb_selected_times[id_action] +=  1
        self.actions_estimates[id_action] += (1/self.nb_selected_times[id_action])*(reward - \
                                                        self.actions_estimates[id_action])
        # chose the action with maximal reward with an epsilon-greedy strategy
        if random.random() < self.epsilon:
            id_action_to_update = random.randint(0, len(self.policy)-1)
        else:
            list_index_val_max = [i for i, v in enumerate(self.actions_estimates) if v == max(self.actions_estimates)]
            id_action_to_update = list_index_val_max[random.randint(0, len(list_index_val_max)-1)]
        # update the policy
        nb_actions = len(self.policy)
        u_p_vec = np.zeros(nb_actions)
        u_p_vec[id_action_to_update] = 1
        self.policy += self.learning_rate * (u_p_vec- self.policy)




##############################################################################################
############################################### Hierarchical Algorithms #######################
##############################################################################################
class HierarContPursuitAgent(BanditAgent):
    """
    Classe abstraite representant un automate hierarchique à base de poursuite avec espace de probabilité continu.
    
    actions : liste contenant l'espace d'action
    pursuit_agents : liste des automates
    nb_levels : nombre de niveau de l'automate hierarchique
    
    Remarque:
    La hiérarchie d'automate est représentée sous forme d'abre. Les actions des automates du niveau les plus bas sont les actions qui affectent
    l'environnement. Les actions des automates du niveau centrale consistent à choisir un des automates adjacents de niveau inférieur.
    La politique est supposé uniforme pour l'algorithme de la poursuite hierarchique.
    """
    def __init__(self, actions, global_policy_updated, learning_rate, epsilon, initial_action_estimate, nb_actions_by_agent):
        """
        Constructeur appelant la fonction qui forme l'arbre d'automate.
        actions : liste représentants l'ensembles des actions agissant sur l'environnement (par ex. les pentes).
        learning_rate : le taux d'apprentissage float [0, 1].
        epsilon : la probabilité de renforcer une action choisie aléatoirement (associé aux estimations des actions - float).
        initial_action_estimate : estimation initiale des actions, la meme pour toutes les actions - float.
        nb_actions_by_agent : le nombre d'action à disposition d'un agent dans la hiérarchie.
        
        Remarque :
            Le nombre de niveau total (nb_levels+1) est le nombre de bits necessaire pour encoder les indices
            des actions en base "nb_actions_by_agent"-aires.
        """
        super().__init__(actions, None)
        self.actions = actions
        self.nb_actions_by_agent = nb_actions_by_agent
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.pursuit_agents, self.last_level_hier, self.last_level_agents, self.nb_levels = self.init_tree_policy(learning_rate, 
                                                                                          epsilon, 
                                                                                          initial_action_estimate, 
                                                                                          nb_actions_by_agent)
        self.global_policy_updated = global_policy_updated
        self.policy = self.update_global_policy(True)
        
#        print("----------------------------------------")
#        print("Nombre de niveaux : ", self.nb_levels)
#        print("Nombre max d'actions par niveau : ", self.nb_actions_by_agent)
#        print("Nombre total d'actions : ", len(actions))
        
        
    def update_global_policy(self, first_update = False):
        """
        Initialise la politique globale.
        """
        if first_update is True:
            self.pursuit_agents.get_data()[1] = 1.0
        stack = []
        stack.append(self.pursuit_agents)
        while len(stack) != 0:# Parcours en profondeur de la hierarchie
            cur_node = stack.pop()
            ag_node = cur_node.get_data()[0]
            pr_cur_node = cur_node.get_data()[1]
            children = cur_node.get_children()
            if children is not None:
                for id_ch in range(len(children)):
                    children[id_ch].get_data()[1] = pr_cur_node * ag_node.policy[id_ch] # Calcul la probabilité de selection de l'agent
                    stack.append(children[id_ch])
        # Construire la politique globale puis la retourner
        global_policy = [node_ac.get_data()[1] for node_ac in self.last_level_hier]
        return global_policy
                
    
    def init_tree_policy(self, learning_rate, epsilon, initial_action_estimate, nb_actions_by_agent):
        """
        Initialisation de l'automate hiéarchique avec la politique initiale.
        
        nb_actions_by_agent: nombre "maximal" d'actions par agent dans la hiérarchie.
        Return :
        cur_top_level : liste de l'agent à la racine de la hiérarchie
        nb_levels : nombre de niveau de la hiérachie d'agents (hors actions affectant l'environnement - seulement les agents)
        Remarques:
        La politique de dépar de l'agent est uniforme. L'estimation intiale est la meme pour tous les agents de la hierarchie.
        key : (indice relatif de l'agent chez son parent = indice de l'agent dans son niveau mod (nb de fils par noeud)
                , l'indice de l'action de niveau le plus bas la plus petite disponible à l'agent)
        indice de l'agent de son niveau = (nombre d'agents de niveau inférieur avant lui) mod (nb de fils par noeud)
        """
        # Initialisation des variables
        cur_top_level = [Node(data=[self.actions[id_ac], None], key = (id_ac%nb_actions_by_agent, id_ac), 
                              children=None, parent=None) for id_ac in range(len(self.actions))]
        next_top_level, nb_levels =  [], 0
        last_level_hier, last_level_agents = cur_top_level, None
        # Création de chaque niveau de l'automate de bas de haut, on rentre au moins une fois dans cette boucle.
        while len(cur_top_level) > 1:
            # Incrementation du nombre de niveau crées et initialisation du prochain de niveau à construire dans l'arbre
            nb_levels += 1
            next_top_level = []
            # Construction du prochain niveau (next_top_level), noeud par noeud
            for id_ac in range(0, len(cur_top_level), nb_actions_by_agent):
                # Indices des fils du prochaine noeud crées
                id_lim = min(id_ac+nb_actions_by_agent, len(cur_top_level))
                actions = cur_top_level[id_ac:id_lim] # Les actions de l'agent sont les fils
                # Creation du prochain noeud du niveau prochain
                policy = np.array([1.0/len(actions) for i in range(len(actions))]) # Politique uniforme
                actions_estimates_per_agent = np.array([initial_action_estimate for i in range(len(actions))])
                agent = PursuitAgent(actions, policy, learning_rate, epsilon, actions_estimates_per_agent)
                node = Node(data = [agent, None], 
                            key = (int(id_ac/nb_actions_by_agent)%nb_actions_by_agent , 
                                   min([ac.get_key()[1] for ac in actions])), 
                            children = actions, parent = None)
                for act in actions: # Mise à jour du parent du noeud de niveau inférieure
                    act.set_parent(node)
                # Ajout du nouveau noeud crée
                next_top_level.append(node)
            # Mise à jour du niveau le plus haut de l'arbre
            cur_top_level = next_top_level
            if nb_levels==1:
                last_level_agents = cur_top_level
        
        if len(cur_top_level) != 1:
            raise Exception("L'aborescence d'agents a plusieurs ou aucune racine.")
        
        return cur_top_level[0], last_level_hier, last_level_agents, nb_levels
    
    
    def reset_agent(self, initial_policy, initial_hyperparams):
        """
        reinitialise la politique apres une execution
        Remarque:
        La valeur par défaut de intial_actions_estimates est de 0.0.
        """
        
        actions_estimates = return_initial_actions_estimates(initial_hyperparams, "Hier_cont_pursuit", len(self.policy))
        
        self.pursuit_agents, self.last_level_hier, self.last_level_agents, self.nb_levels = self.init_tree_policy(self.pursuit_agents.get_data()[0].learning_rate, 
                                                                                          self.pursuit_agents.get_data()[0].epsilon, 
                                                                                          actions_estimates, 
                                                                                          self.nb_actions_by_agent)
        self.policy = self.update_global_policy(True)


#    def __repr__(self):
#        """
#        Parcours l'arborescence de l'agent et affiche les c aracteristiques de l'agent tel que
#        """
#        cur_node = self.pursuit_agents
#        while cur_node.get_children() is not None:
#            # On récupère l'agent courant
#            agent = cur_node.get_data()[0]
#            # Délimination de l'espace des actions (agent des valeurs)
#            if id_end is None: 
#                id_end_cur_ag = id_end
#            else:
#                id_end_cur_ag = max(len([ac.get_key()[1] for ac in agent.actions if ac.get_key()[1] <= id_end]) - 1, 0)
#            # Choix de l'action et sélection potentiel du prochain agent
#            id_action = agent.chose_action(id_end_cur_ag)
#            cur_node = cur_node.get_children()[id_action]
#        return np.searchsorted(self.actions, cur_node.get_data()[0])
    

    def chose_action(self, id_end=None):
        """
        Choisit l'action en échantillonant recursivement le vecteur de probabilité.
        Le choix de l'action définit un parcours partant de la racine à une feuille dans la hiérarchie.
        Retourne: l'indice de l'action au le dernire niveau de la hiérarchie (avec elegage à partir de "id_end").
        """
        cur_node = self.pursuit_agents
        while cur_node.get_children() is not None:
            # On récupère l'agent courant
            agent = cur_node.get_data()[0]
            # Délimination de l'espace des actions (agent des valeurs)
            if id_end is None: 
                id_end_cur_ag = id_end
            else:
                id_end_cur_ag = max(len([ac.get_key()[1] for ac in agent.actions if ac.get_key()[1] <= id_end]) - 1, 0)
            # Choix de l'action et sélection potentiel du prochain agent
            id_action = agent.chose_action(id_end_cur_ag)
            cur_node = cur_node.get_children()[id_action]
        return np.searchsorted(self.actions, cur_node.get_data()[0])
            
    
    def update_policy(self, id_glob_action, reward):
        # Mise à jour des données (d'estimations) relatifs à l'action qui avait été choisie
        last_sel_agent = self.last_level_hier[id_glob_action].get_parent().get_data()[0]
        id_ac_agent = self.last_level_hier[id_glob_action].get_key()[0]
        
        last_sel_agent.nb_selected_times[id_ac_agent] +=  1
        last_sel_agent.actions_estimates[id_ac_agent] += (1/last_sel_agent.nb_selected_times[id_ac_agent])*(reward - \
                                                        last_sel_agent.actions_estimates[id_ac_agent])
        
        # Chose the action with maximal reward with an epsilon-greedy strategy
        if random.random() < self.epsilon:
            id_glob_action_to_update = random.randint(0, len(self.actions)-1)
        else:
            actions_estimates = [e for a in self.last_level_agents for e in a.get_data()[0].actions_estimates]
            list_index_val_max = [i for i, v in enumerate(actions_estimates) if v == max(actions_estimates)]
            id_glob_action_to_update = list_index_val_max[random.randint(0, len(list_index_val_max)-1)]
        
        # Update la politique en commençant par l'agent le plus en bas
        ac_node = self.last_level_hier[id_glob_action_to_update]
        ag_node = ac_node.get_parent()
        # Mise à jour iterativement de tous le chemin menant à la meilleure action actuelle
        while ag_node is not None:
            # Mise à jour de 
            ag = ag_node.get_data()[0]
            u_p_vec = np.zeros(len(ag.actions))
            id_ac_up = ac_node.get_key()[0] # clé (indice) relatif
            u_p_vec[id_ac_up] = 1
            ag.policy += ag.learning_rate * (u_p_vec - ag.policy)
            # Monter d'un pas en haut de l'arbre pour traiter la politique du père de l'agent actuel
            ac_node = ag_node
            ag_node = ag_node.get_parent()
        
        if self.global_policy_updated:
            self.policy = self.update_global_policy()




###########################################################################################################
################################ Helper functions to create agents ########################################
###########################################################################################################
def return_initial_actions_estimates(initial_hyperparams, ag_type, nb_actions):
    """
        Retourne les estimations initiales des actions suivant le type d'agent utilisé.
        Rq :
        - Dans le cas ou on utilise un agent non hiérarchiqe, une liste est retournée.
        - Dans le cas ou l'agent est "Hier_cont_pursuit" le nombre d'action varie suivant les agents dans la hiérarchie.
          Pour cette raison une seule valeur seulement est retournée.
    """
    if ag_type == "Hier_cont_pursuit":
        if "initial_actions_estimates" in initial_hyperparams:
            initial_action_estimate = initial_hyperparams["initial_actions_estimates"]
        else:
            initial_action_estimate = 0.0
    
    elif ag_type == "non_hier":
        if "initial_actions_estimates" in initial_hyperparams:
            if isinstance(initial_hyperparams["initial_actions_estimates"], list):
                initial_action_estimate = copy.copy(initial_hyperparams["initial_actions_estimates"])

            elif isinstance(initial_hyperparams["initial_actions_estimates"], int) or isinstance(initial_hyperparams["initial_actions_estimates"], float):
                initial_action_estimate = [initial_hyperparams["initial_actions_estimates"] for i in range(nb_actions)]
            
            else:
                raise Exception("Initial estimate provided unrecongnized.")

        else:
            initial_action_estimate = [0.0 for i in range(nb_actions)]
    
    else:
        raise Exception("Agent type "+ag_type+" is not recongnized.")
    
    return initial_action_estimate



def return_initial_policy(policy, nb_actions):
    """
        Retourne la politique initiale, plusieurs formats sont possibles.
    """
    if isinstance(policy, str) and policy == "uniform": # politique uniforme
        initial_policy = np.array([1/nb_actions for i in range(nb_actions)])
        
    elif isinstance(policy, int): # policy int => indice de l'action ayant un proba 1
        initial_policy = np.array([0.0 for i in range(nb_actions)])
        initial_policy[policy] = 1.0
        
    elif isinstance(policy, dict): # dict contenant les probas s
        initial_policy = np.array([0.0 for i in range(nb_actions)])
        for key in policy.keys():
            initial_policy[key] = policy[key]
        
    elif isinstance(policy, list) or isinstance(policy, np.ndarray): # liste
        initial_policy = policy
    
    elif policy is None: # Pas de politique (pour les agent basés sur les valeurs)
        initial_policy = None
    
    else:
        raise Exception("Initial policy "+policy+" not recongnized.")
    
    return initial_policy


def return_agent(ag_type, actions, initial_policy, lr, eps = None, opt_params = None):
    """
    Retourne un BanditAgent de type "ag_type".
    
    ag_type : string contenant le type d'agent BanditAgent
    actions : liste des actions disponible à l'agent en creation
    lr : taux d'apprentissage de l'agent
    eps : taux d'exploration de l'agent
    opt_params : dictionnaire contenant les paramètres optionnels de l'agent
    
    Return
    ag : objet de classe implementant BanditAgent correspondant à ag_type
    """
    ag = None
    
    # Creation de l'agent suivant la valeu de ag_type
    if ag_type == "random":
        ag = RandomAgent(actions,  
                           policy = initial_policy)
        
    if ag_type == "Eps_exp0":
        ag = EpsilonGreedy(actions, 
                           epsilon = eps, 
                           actions_estimates = return_initial_actions_estimates (opt_params, "non_hier", len(actions)), 
                           exploration = 0)
    
    elif ag_type == "Eps_exp1":
        ag = EpsilonGreedy(actions, 
                           epsilon = eps, 
                           actions_estimates = return_initial_actions_estimates (opt_params, "non_hier", len(actions)), 
                           exploration = 1)
    
    elif ag_type == "Eps_exp2":
        ag = EpsilonGreedy(actions,
                           epsilon = eps, 
                           actions_estimates = return_initial_actions_estimates (opt_params, "non_hier", len(actions)), 
                           exploration = 2)
    
    elif ag_type == "UCB":
        ag = UCBAgent(actions, 
                      epsilon = eps, 
                      actions_estimates = return_initial_actions_estimates (opt_params, "non_hier", len(actions)))
    
    elif ag_type == "LRI":
        ag = LRAgent(actions = actions, 
                 policy = initial_policy,
                 learning_rate = lr, 
                 learning_rate_r0 = None, 
                 ag_type = "LRI")
    
    elif ag_type == "LRP":
        ag = LRAgent(actions = actions, 
                     policy = initial_policy, 
                     learning_rate = lr, 
                     learning_rate_r0 = eps, 
                     ag_type = "LRP")
        
    elif ag_type == "Pursuit":
        ag = PursuitAgent(actions = actions, 
                          policy = initial_policy, 
                          learning_rate = lr, 
                          epsilon = eps,
                          actions_estimates = return_initial_actions_estimates (opt_params, "non_hier", len(actions)))
    
    elif ag_type == "Hier_cont_pursuit":
        ag = HierarContPursuitAgent(actions = actions,
                                         global_policy_updated = True,
                                         learning_rate = lr, 
                                         epsilon = eps, 
                                         initial_action_estimate = return_initial_actions_estimates (opt_params, "Hier_cont_pursuit", len(actions)),
                                         nb_actions_by_agent = opt_params["nb_actions_by_agent"])
    
    elif ag_type == "Exp3":
        ag = Exp3Agent(actions, 
                       epsilon= eps, 
                       actions_estimates = return_initial_actions_estimates (opt_params, "non_hier", len(actions)))
    return ag



###################################################################################################################################
##################################################   Action subspace functions   ##################################################
###################################################################################################################################
def chose_action_actionsubspace (rl_agent, action_subspace, ag_type):
    if ag_type == "Hier_cont_pursuit":
        print("Hierachical pusuite algorithm is not supported.")
        sys.exit()

    if ag_type in POLICY_BASED_TYPE_AGENTS or ag_type in MIXED_TYPE_AGENTS:
        # Replace total action space and policy with subaction space given in entry and its corresponing subpolicy
        actions, policy = rl_agent.actions, rl_agent.policy
        rl_agent.actions = action_subspace
        rl_agent.policy = [policy[rl_agent.actions_to_inds[action]] for action in action_subspace]
        sum_proba_actionsubspace = sum(rl_agent.policy)
        rl_agent.policy = np.array([p/sum_proba_actionsubspace for p in rl_agent.policy])
        # Chose action using subpolicy
        ind_action_acsubspace = rl_agent.chose_action()
        # Replace original action space and policy
        rl_agent.actions, rl_agent.policy = actions, policy # !!!!! (quik hack) peut etre qua ça va changer !!!!

    elif ag_type in VALUE_BASED_TYPE_AGENTS:
        # Replace total action space and estimates with subaction space given in entry and their corresponing estimates
        actions, actions_estimates = rl_agent.actions, rl_agent.actions_estimates
        rl_agent.actions = action_subspace
        rl_agent.actions_estimates = [actions_estimates[rl_agent.actions_to_inds[action]] for action in action_subspace]
        # Chose action using actions estimates of the actions in the subaction space
        ind_action_acsubspace = rl_agent.chose_action()
        # Replace original action space and policy
        rl_agent.actions, rl_agent.actions_estimates = actions, actions_estimates # !!!!! (quik hack) peut etre qua ça va changer !!!!
    
    else:
        print("Agent type not recognized.")
        sys.exit()

    return ind_action_acsubspace




###################################################################################################################################
##################################################   Test functions   #############################################################
###################################################################################################################################
def basic_test_rl_agents(agent_types,
                         actions,
                         expected_rewards,
                         nb_interactions,
                         lr_ls,
                         eps_ls,
                         initial_policy = "uniform",
                         opt_params = None):
    dict_res = {}
    # For each agent create it and test it
    for i in range(len(agent_types)):
        # Fetch agent type
        ag_type = agent_types[i]
        # Agent creation
        ag = return_agent(ag_type = ag_type, 
                          actions = actions, 
                          initial_policy = initial_policy, 
                          lr = lr_ls[i], 
                          eps = eps_ls[i], 
                          opt_params = opt_params)
        # Empirical policy
        emp_policy = [0 for _ in range(len(actions))]
        # Test the agent with 'nb_interactions' learning step
        for _ in range(nb_interactions):
            id_ac = ag.chose_action()
            r = float(random.random() < expected_rewards[id_ac])
            ag.update_policy(id_ac, r)
            emp_policy[id_ac] += 1
        # Update emprical policy
        dict_res[ag_type] = [count/sum(emp_policy) for count in emp_policy]
    return dict_res




###################################################################################################################################
##################################################   Main   #############################################################
###################################################################################################################################
def main():
    test_names = {"basic_test_rl_agents"}
    test_name = "basic_test_rl_agents"
    if test_name == "basic_test_rl_agents":
        nb_actions = 3
        ag_types = ["random", "LRI", "LRP", "Eps_exp0", "Eps_exp1", "Eps_exp2",  
                    "Pursuit", "Hier_cont_pursuit", "Exp3", "UCB"]
        lr_ls = [0.01 for _ in range(len(ag_types))]
        eps_ls = [0.1 for _ in range(len(ag_types) - 1)]+[math.sqrt(2)]
        emp_policy = basic_test_rl_agents(agent_types = ag_types,
                                            actions = [i for i in range(nb_actions)],
                                            expected_rewards = [0.2, 0.5, 0.2],
                                            nb_interactions = 10000,
                                            initial_policy = "uniform",
                                            lr_ls = lr_ls,
                                            eps_ls = eps_ls,
                                            opt_params = dict({"nb_actions_by_agent":2}))
        print(emp_policy)

if __name__ == "__main__":
    main()
    
