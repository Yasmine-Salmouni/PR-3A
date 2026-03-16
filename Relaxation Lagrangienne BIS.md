# Relaxation Lagrangienne BIS

Je vais modifier mon modele pour suivre l’article: 

Solving Manufacturing-Remanufacturing System Production Planning Problems Using Lagrangian Relaxation

Etapes de relaxation lagrangienne: Fisher, M.L. (1985), "An Applications Oriented Guide to Lagrangian Relaxation", publié dans la revue *Interfaces*.

---

## Les sous problèmes

Pour ton projet, la section 4.2.2 correspond au moment où tu vas **réorganiser mathématiquement** ta grande équation pour faire apparaître tes propres sous-problèmes indépendants.
Voici comment démontrer cette décomposition de manière exacte.
**A. La réorganisation mathématique**
Reprenons ton grand modèle relâché (le Lagrangien) de l'étape précédente :$L(R, O, \lambda) = \sum_{c, dc, p, dp} (R_{c,dc,p,dp} - O_{p,dp,c}) + \sum_{c, dc} \lambda_{c,dc} \left( stock\_proj\_chute_{c,dc} - \sum_{p, dp} R_{c,dc,p,dp} \right)$
Pour faire la "Décomposition" (Section 4.2.2), on va développer cette formule et regrouper les termes :
1. On sépare la partie fixe du stock : $\sum (\lambda \times stock)$.
2. On regroupe tout ce qui concerne les variables de décision ($R$ et $O$) pour chaque production $(p, dp)$.
Ton Lagrangien se réécrit donc EXACTEMENT comme ceci :$L(\lambda) = \underbrace{\sum_{c, dc} (\lambda_{c,dc} \times stock\_proj\_chute_{c,dc})}_{\text{Valeur fixe (Constante pour un } \lambda \text{ donné)}} + \underbrace{\sum_{p \in P} \sum_{dp \in DP} \left[ \sum_{c, dc} \left( (1 - \lambda_{c,dc}) R_{c,dc,p,dp} - O_{p,dp,c} \right) \right]}_{\text{La somme de tes sous-problèmes indépendants}}$
**B. Tes sous-problèmes (L'équivalent de ses 5 sous-problèmes)**
L'auteure a 5 sous-problèmes liés à la nature de ses pièces. Toi, grâce à la réorganisation ci-dessus, tu vois que ton problème s'est décomposé en **une multitude de sous-problèmes : un pour chaque production $(p, dp)$**.
Pour une production $(p, dp)$ précise (par exemple "Fabriquer le pneu P1 le jour J1"), la décision n'est plus liée aux autres productions. Le sous-problème local consiste uniquement à **maximiser** la parenthèse interne :$\max \left[ (1 - \lambda_{c,dc}) R_{c,dc,p,dp} - O_{p,dp,c} \right]$

**C. Ce que tu dois faire en pratique (Ton code) :**

Appliquer la section 4.2.2 signifie que dans ton algorithme, tu as maintenant le droit mathématique de faire une simple **boucle isolée**:

Pour chaque production p:
Pour chaque date dp:
# Résoudre le sous-problème(p, dp) en ignorant totalement le reste de l'usine !

---

## Retirer l’étape B avec l’heuristique

---

## La méthode du sous gradient

Formules utilisées: Fisher, M.L. (1985), "An Applications Oriented Guide to Lagrangian Relaxation", publié dans la revue *Interfaces*.

**Le point de départ :** On commence avec une valeur arbitraire pour $\lambda^0$

**Étape D.4 : La règle stricte du paramètre $\alpha^k$**
Pour coller parfaitement à la méthodologie de l'auteure :
1. À la toute première itération ($k=1$), tu fixes $\alpha = 2.0$.
2. Si, pendant **5 itérations consécutives**, ta Borne Duale $Z_{dual}$ ne diminue pas (c'est-à-dire qu'elle ne s'améliore pas vers ta cible), tu dois faire : $\alpha = \alpha / 2$ (donc il passera à 1.0, puis 0.5, puis 0.25, etc.). (Note : Le chiffre de 5 itérations est celui utilisé dans le code source Python de l'auteure fourni en annexe de son mémoire ).

---

## **L’algorithme final (Le calque exact de l'article)**

Voici comment ton code complet doit être structuré désormais, en traduisant les 9 étapes pour ton problème de **maximisation** :
**--- INITIALISATION ---**
• **Step 1 & 2 :** Trouver une première solution valide (même mauvaise) pour avoir une Borne Supérieure  de départ . Crée un petit plan de production de départ très simple qui respecte parfaitement tes stocks (par exemple, n'alimenter que les 3 premières productions de la liste). Calcule son vrai gain et enregistre-le dans la variable **$Z_{best\_primal}$** (C'est ta Borne Inférieure à battre).
• **Step 3 :** Initialise $k = 1$. Fixe tous tes multiplicateurs $\lambda_{c,dc} = 0.1$ (comme le fait l'auteure dans son code). Fixe le paramètre du pas $\alpha = 2.0$.
**--- LA BOUCLE (tant que $k \le 300$ itérations) ---**
• **Step 4 & 5 (Ton Étape A de Score) :**
Pour chaque production $(p, dp)$, calcule le `Score = [(1 - lambda) * reincorpo_max] - 1` pour toutes les chutes. Prends le meilleur score positif.
Sauvegarde les décisions brutes : $R_{brut}$ et $O_{brut}$.
• **Step 6 (Borne Duale) :**
Calcule ton Lagrangien : $Z_{dual}^{(k)} = \sum (\lambda \times stock) + \sum (\text{Scores Max Positifs})$.
• **Step 7 (Test de Faisabilité strict) :**
Prends tes décisions brutes $R_{brut}$ et somme-les par type de chute.
*Question :* Est-ce que $\sum R_{brut} \le stock\_proj\_chute$ pour **absolument toutes** les chutes ?
    ◦ **OUI :** Ta solution est "naturellement" valide ! Calcule son vrai gain global $\sum(R_{brut} - O_{brut})$. Si ce gain est $> Z_{best\_primal}$, alors met à jour : $Z_{best\_primal} = \text{ce nouveau gain}$, et sauvegarde ce plan de production comme le meilleur trouvé.
    ◦ **NON :** Ta solution est irréaliste. Ne fais rien, ne la répare pas, garde l'ancien $Z_{best\_primal}$.
• **Step 8 (Sous-gradient et Mise à jour) :**
    ◦ *Arrêt :* Si $(Z_{dual}^{(k)} - Z_{best\_primal}) \le \epsilon$ (ex: 0.01), ARRANGE-TOI POUR STOPPER LA BOUCLE.
    ◦ *Mise à jour de $\alpha$ :* Si $Z_{dual}$ ne s'est pas amélioré (n'a pas baissé) depuis 5 itérations, divise $\alpha$ par 2 ($\alpha = \alpha / 2$).
    ◦ *Calcul des erreurs $g$ :* Pour chaque chute, calcule la violation : $g = \sum R_{brut} - stock$.
    ◦ *Calcul du pas $t_k$ :* $t_k = \frac{\alpha \times (Z_{dual}^{(k)} - Z_{best\_primal})}{\sum (g)^2}$
    ◦ *Nouveaux prix :* $\lambda_{c,dc} = \max(0, \lambda_{c,dc} + t_k \times g)$
    ◦ Passe à $k = k + 1$ et recommence au Step 4.
**--- FIN ---**
• **Step 9 :** Affiche le plan de production associé à ton dernier $Z_{best\_primal}$.