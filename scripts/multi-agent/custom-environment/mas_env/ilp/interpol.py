import pandas as pd

def interpolate(path, delta, target, day_of_year=None):
    delta = int(delta)/60
    target = target/60
    
    if(delta < target):
        return None
    
    df = pd.read_csv(path)
    df['period_end'] = pd.to_datetime(df['period_end'])

    df = df.set_index('period_end')

    start = df.index.min()
    end = df.index.max()
    new_index = pd.date_range(start=start, end=end, freq=f'{target}min')

    df_interpol = df.reindex(new_index)
    df_interpol['dni'] = round(df_interpol['dni'].interpolate(method='linear'), 2)
    df_interpol['ghi'] = df_interpol['ghi'].interpolate(method='linear')
    
    if day_of_year is not None:
        df_interpol = df_interpol[df_interpol.index.dayofyear == day_of_year]

    return df_interpol
