import numpy as np
# wykaz czasu wykoanania danych funkcji

def test4():
    time_log_file = open("execution_time_log.txt", "r")

    time_list = []

    for line in time_log_file:
        name = line.split()[0]
        time = float(line.split()[1])

        #print(name, time)
        log_dict = dict(name =name, time = time)
        time_list.append(log_dict)
    
    time_log_file.close()

    #merge dict with the same name, and calculate average time
    time_dict = {}
    for log in time_list:
        if log['name'] in time_dict:
            time_dict[log['name']].append(log['time'])
        else:
            time_dict[log['name']] = [log['time']]

    for key in time_dict:
        time_dict[key] = np.mean(time_dict[key])
        time_dict[key] = round(time_dict[key], 4)

    for key in time_dict:
        print(key, time_dict[key])





if __name__ == '__main__':
    test4()
