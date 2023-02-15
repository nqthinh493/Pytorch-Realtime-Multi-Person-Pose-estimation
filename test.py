import pandas as pd
import os
if os.path.exists('filename.csv'):
    pass
else:
    pd.DataFrame().to_csv("filename.csv")

t = pd.read_csv('filename.csv')
s = pd.DataFrame([[1.3, 9]], columns = ["A", "B"])
s2 = pd.DataFrame([[9]], columns = ["B"])
t= pd.concat([t, pd.DataFrame(s)], axis=0, ignore_index= True)
t= pd.concat([t, pd.DataFrame(s2)], axis=0, ignore_index= True)
t.to_csv("filename.csv", index= False)
print(t)

def write_loss_csv(csv_path, loss, loss_states):
    if os.path.exists(csv_path):
        pass
    else:
        pd.DataFrame().to_csv(csv_path)
        
    df = pd.read_csv(csv_path)
    total_loss = pd.DataFrame([[loss.val, loss.avg]], columns = ["Total Loss", "Loss avg"])

    df = pd.concat([df, pd.DataFrame(total_loss)], axis=0, ignore_index= True)
    for name, value in loss_states.items():
        loss_stages = pd.DataFrame([[value.val, value.avg]], columns = [f"{name} value", f"{name} svg"])
        df = pd.concat([df , pd.DataFrame(loss_stages)], axis=0, ignore_index = True)
    df.to_csv(csv_path, index= False)


