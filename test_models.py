from training import main, get_args
from pathlib import Path
import shutil

if __name__ == "__main__":
    from utils import io
    import numpy as np
    import pandas as pd

    input_path=Path("/home/smonaco/rainfall-prediction/data/24h_10mmMAX_OI")
    case_study = input_path.stem
    seed = 42
    case_study_max, available_models, _, _, test_dates, indices_one, indices_zero, _, _, _ = io.get_casestudy_stuff(
                input_path, n_split=1, case_study=case_study, ispadded=True,
                seed=seed)
    extreme_events = pd.concat([pd.read_csv(exev, header=None) for exev in input_path.glob('*extremeEvents.csv')])[0].values
    not_extreme_events = np.array([date for date in test_dates if date not in extreme_events])
    all_dates = np.concatenate([extreme_events, not_extreme_events])
    X, Y, _, _ = io.load_data(input_path, all_dates, case_study_max, 
                                                        indices_one, indices_zero, available_models)

    # for model in ['unet', 'sde_unet', 'ensemble_unet', 'mcd_unet']:
    #     print(f"""
    #           \n\n================================
    #           >> Testing model {model}\n""")
    #     args = get_args(['--network_model', model, '-e', '1'])
    #     output_path = Path('lightning_logs') / f'{args.network_model}' / f"split_{args.n_split}"
    #     assert not output_path.exists(), f"Output path {output_path} already exists"
    #     main(args)
    #     shutil.rmtree(output_path)