from pathlib import Path
from ultralytics import YOLO
from smac import MultiFidelityFacade as MFFacade
from smac import Scenario
from smac.facade import AbstractFacade
from smac.intensifier.hyperband import Hyperband
from smac.intensifier.successive_halving import SuccessiveHalving
from ConfigSpace import (
    ConfigurationSpace,
    Float,
)

def objective_function(config , budget , seed):
    epochs = int(budget)
    patience = epochs
    model = YOLO("yolov8n.pt")
    print(config)
    results = model.train(data="screw_data.yaml",
                          epochs=epochs, batch=16, optimizer= "AdamW" ,imgsz=1664 , patience=patience, device='cuda:0' , **config ,project="samc_tune" )
  
    
    return 1 - results.results_dict['metrics/mAP50(B)']



if __name__ == "__main__":

    

    cs = ConfigurationSpace()

    cs = ConfigurationSpace()

    lr0 = Float("lr0", (1e-5, 1e-1), log=True, default=1e-2)
    lrf = Float("lrf", (0.0001, 0.1), log=True, default=0.001)
    momentum = Float("momentum", (0.7, 0.98), default=0.9)
    weight_decay = Float("weight_decay", (0.0, 0.001), default=5e-4)
    warmup_epochs = Float("warmup_epochs", (0.0, 5.0), default=0.0)
    warmup_momentum = Float("warmup_momentum", (0.0, 0.95), default=0.0)
    box = Float("box", (1.0, 20.0), default=1.0)
    cls = Float("cls", (0.2, 4.0), default=1.0)
    dfl = Float("dfl", (0.4, 6.0), default=1.0)
    hsv_h = Float("hsv_h", (0.0, 0.1), default=0.0)
    hsv_s = Float("hsv_s", (0.0, 0.9), default=0.0)
    hsv_v = Float("hsv_v", (0.0, 0.9), default=0.0)
    degrees = Float("degrees", (0.0, 45.0), default=0.0)
    translate = Float("translate", (0.0, 0.9), default=0.0)
    scale = Float("scale", (0.0, 0.95), default=0.0)
    shear = Float("shear", (0.0, 10.0), default=0.0)
    perspective = Float("perspective", (0.0, 0.001), default=0.0)
    flipud = Float("flipud", (0.0, 1.0), default=0.0)
    fliplr = Float("fliplr", (0.0, 1.0), default=0.0)
    mosaic = Float("mosaic", (0.0, 1.0), default=0.0)
    mixup = Float("mixup", (0.0, 1.0), default=0.0)
    copy_paste = Float("copy_paste", (0.0, 1.0), default=0.0)

    cs.add_hyperparameters([lr0, lrf, momentum, weight_decay, warmup_epochs, warmup_momentum, box, cls, dfl,
                            hsv_h, hsv_s, hsv_v, degrees, translate, scale, shear, perspective,
                            flipud, fliplr, mosaic, mixup, copy_paste])


    scenario = Scenario(
        configspace=cs,
        output_directory=Path("smac"),
        deterministic=True,
        walltime_limit=18000,  
        n_trials=15,  
        min_budget=10, 
        max_budget=100,
    )

    initial_design = MFFacade.get_initial_design(scenario, n_configs=5)


    smac = MFFacade(scenario, objective_function , initial_design=initial_design,
            overwrite=True,)
    
    incumbent = smac.optimize()
    print(incumbent)

    incumbent_cost = smac.validate(incumbent)
