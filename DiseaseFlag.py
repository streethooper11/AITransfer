from enum import Flag, auto


# Flags for diseases; if the result of bitwise AND with HasDisease is 0, then no disease is detected
class Diseases(Flag):
    Atelectasis = auto()
    Cardiomegaly = auto()
    Consolidation = auto()
    Edema = auto()
    Effusion = auto()
    Emphysema = auto()
    Fibrosis = auto()
    Hernia = auto()
    Infiltration = auto()
    Mass = auto()
    Nodule = auto()
    Pleural_Thickening = auto()
    Pneumonia = auto()
    Pneumothorax = auto()
    HasDisease = Atelectasis | Cardiomegaly | Consolidation | Edema | Effusion | Emphysema | Fibrosis | \
        Hernia | Infiltration | Mass | Nodule | Pleural_Thickening | Pneumonia | Pneumothorax
