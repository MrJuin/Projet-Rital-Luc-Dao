# -*- coding: utf-8 -*-
import numpy as np
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
class Equilibrage:
    def equilibrate_court(datax,datay,f1 = 0.5, f2 = 1):
        if f1 != None:
            part1 = RandomUnderSampler(sampling_strategy=f1)
            datax, datay = part1.fit_resample(datax,datay)
        if f2 == None:
            return datax, datay
        part2 = RandomOverSampler(sampling_strategy=f2)
        return part2.fit_resample(datax,datay)
    
    def equilibrate_long(datax,datay,f1 = 0.5, f2 = 1):
        if f1 != None:
            part1 = RandomUnderSampler(sampling_strategy=f1)
            datax, datay = part1.fit_resample(datax,datay)
        if f2 == None:
            return datax, datay
        part2 = SMOTETomek(sampling_strategy=f2)
        return part2.fit_resample(datax,datay)
        