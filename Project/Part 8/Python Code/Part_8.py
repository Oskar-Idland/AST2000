from ast2000tools.relativity import RelativityExperiments  

seed = 36874
planet_index = 1
Rel_Exp = RelativityExperiments(seed) 
Rel_Exp.spaceship_duel(planet_index) # Generates XML files 
Rel_Exp.spaceship_race(planet_index)
Rel_Exp.antimatter_spaceship(planet_index)