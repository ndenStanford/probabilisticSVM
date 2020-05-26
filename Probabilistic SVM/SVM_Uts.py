import numpy as np
from sklearn import preprocessing
from sklearn import svm
# generate training samples
def generate_samples(data_array,num_samples,sample_size):
    for i in range(num_samples):
        np.random.seed(seed=i)
        samples=np.round_(data_array[np.random.choice(data_array.shape[0], size=sample_size, replace=False), :],decimals=1)
        filename='generated_samples\\sample'+str(i)+'.txt'
        np.savetxt(filename,samples)
        # write file header
        with open(filename,"r+") as f:
            a=f.read()
            line_prepend='wells\n3\nX\nY\nclassLitho\n'
        with open(filename, "w+") as f:    
            f.write(line_prepend + a)
            
def fitSVM(data,sample_ind,sample_size,c,gamma):
    #contour plot
    data_array=np.array(data)[:,[0,1,3]]
    # generate training samples
    np.random.seed(sample_ind)
    samples=data_array[np.random.choice(data.shape[0], size=sample_size, replace=False), :]                
    lab_enc = preprocessing.LabelEncoder()
    encoded = lab_enc.fit_transform(samples[:,-1])
    model_truth=svm.SVC(C=c,kernel='rbf', gamma=gamma)
    model_truth.fit(samples[:,:-1], encoded)
    return model_truth,samples 

def generate_average_bucket_all(sample_ind,ground_truth,pi_score,n_bucket):
    # check confidence with the actual count
    p=sample_ind
    average_bucket=[]
    average_bucket_truth=[]
    for j in range(n_bucket):
        b=np.array([ground_truth[i] for i in range(len(pi_score)) \
                    if (pi_score[i] > (j/n_bucket) and pi_score[i] < ((j+1.0)/n_bucket))])
        #print('b.shape = ',b.shape)
        average_bucket.append(np.sum(b)/len(b))
        b2=np.array([pi_score[i] for i in range(len(pi_score)) \
                    if (pi_score[i] > (j/n_bucket) and pi_score[i] < ((j+1.0)/n_bucket))])
        #print('b2.shape = ',b2.shape)
        average_bucket_truth.append(np.mean(b2))
    center_bucket=np.linspace(0,1,n_bucket+1)+(1/(2*n_bucket))
    center_bucket=center_bucket[:-1]
    average_bucket_all[:,p]=average_bucket
    return average_bucket_all,center_bucket

def plot_bin_graph(average_bucket_all,center_bucket):    
    average_bucket_mean=np.mean(average_bucket_all,axis=1)
    average_bucket_sd=np.std(average_bucket_all,axis=1)
    plt.figure()
    plt.title('sample_size'+str(sample_size)+' gamma = '+str(gamma)+' C = '+str(c))
    plt.xlabel('calculated probability of classified as 1')
    plt.ylabel('real probability of classified as 1')
    #plt.scatter(center_bucket,average_bucket)
    #plt.scatter(average_bucket_truth,average_bucket)
    plt.plot([0,1],[0,1])
    #plt.plot(center_bucket,average_bucket)
    plt.errorbar(center_bucket,average_bucket_mean,average_bucket_sd, uplims=True, lolims=True) 

'''
Input parameters:
out = array of SVM outputs
target = array of booleans: is ith example a positive example?
prior1 = number of positive examples
prior0 = number of negative examples
Outputs:
A, B = parameters of sigmoid
'''
def sigmoid_training(out,target,prior1,prior0):
    A = 0
    B = np.log((prior0+1)/(prior1+1))
    hiTarget = (prior1+1)/(prior1+2)
    loTarget = 1/(prior0+2)
    lamb = 1e-3
    olderr = 1e300
    #pp = temp array to store urrent estimate of probability of examples
    #set all pp array elements to (prior1+1)/(prior0+prior1+2)
    pp=np.ones(len(out))*(prior1+1)/(prior0+prior1+2)
    count = 0
    #for it = 1 to 100 {
    #for it in range(1,100+1):
    for it in range(1,100+1):
        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        # First, compute Hessian & gradient of error funtion
        # with respet to A & B
        #for i = 1 to len {
        for i in range(len(out)):
            if (target[i]):
                t = hiTarget
                #t=1
            else:
                t = loTarget
                #t=0
            d1 = pp[i]-t
            d2 = pp[i]*(1-pp[i])
            a += out[i]*out[i]*d2
            b += d2
            c += out[i]*d2
            d += out[i]*d1
            e += d1
        # If gradient is really tiny, then stop
        #if (abs(d) < 1e-9 and abs(e) < 1e-9):
        if (abs(d) < 1e-15 and abs(e) < 1e-15):
            break
        oldA = A
        oldB = B
        err = 0
        # Loop until goodness of fit inreases
        while (1):
            det = (a+lamb)*(b+lamb)-c*c
            if (det == 0): # if determinant of Hessian is zero,
                # inrease stabilizer
                lamb *= 10
                continue	
            A = oldA + ((b+lamb)*d-c*e)/det
            B = oldB + ((a+lamb)*e-c*d)/det
            # Now, compute the goodness of fit
            err = 0;
            for i in range(len(out)):
                p = 1/(1+np.exp(out[i]*A+B))
                pp[i] = p
                # At this step, make sure log(0) returns -200
                err -= t*np.log(p)+(1-t)*np.log(1-p)
                #import pdb; pdb.set_trace()
            if (err < olderr*(1+1e-7)):
                lamb *= 0.1
                
                break
            # error did not derease: inrease stabilizer by fator of 10
            # & try again
            lamb *= 10
            if (lamb >= 1e6): # something is broken. Give up
                #import pdb; pdb.set_trace()
                break
        diff = err-olderr
        sale = 0.5*(err+olderr+1)
        if (diff > -1e-3*sale and diff < 1e-7*sale):
            count+=1
        else:
            count = 0
        olderr = err
        if (count == 3):
            break
    return A, B, err

def pi(A,B,fi):
    return 1/(1+np.exp(A*fi+B))

