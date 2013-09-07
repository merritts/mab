def regret(best_arm, arms, arm_choices):
    a = 0.
    for i,j in enumerate(arm_choices):
        a+=(arms[best_arm]-arms[i])*j
    return a