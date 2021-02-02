import numpy as np
import matplotlib.pyplot as plt


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm


def visualize_dict(dic):
    for i, m in dic.items():
        print(i, m.shape)
        print(m)

def sigmoid(Z):
   return 1 / (1 + np.exp(-Z))

def initialize_parameters(X, layers, way = 'default_rnd'):
    parameters = dict()

    if way == 'default_rnd':
        parameters["W1"] = np.random.rand(layers[0], X.shape[0])
        parameters["b1"] = np.zeros((layers[0], 1))
        for i in range(1, len(layers)):
            parameters["W" + str(i + 1)] = np.random.rand(layers[i], layers[i - 1])
            parameters["b" + str(i + 1)] = np.zeros((layers[i], 1))

    elif way == 'xavier_rnd':
        parameters["W1"] = np.random.rand(layers[0], X.shape[0]) * np.sqrt(1/X.shape[0])
        parameters["b1"] = np.zeros((layers[0], 1))
        for i in range(1, len(layers)):
            parameters["W" + str(i + 1)] = np.random.rand(layers[i], layers[i - 1]) * np.sqrt(1 / layers[i - 1])
            parameters["b" + str(i + 1)] = np.zeros((layers[i], 1))

    return parameters


def forward_propagation(X, parameters, activations):

    L = int(len(parameters)/2)
    cache = dict()
    cache['Z1'] = np.dot(parameters['W1'], X) + parameters['b1']
    cache['A1'] = perform_activation(cache['Z1'], activations[0])

    for i in range(1, L):
        cache['Z' + str(i+1)] = np.dot(parameters['W' + str(i+1)], cache['A' + str(i)]) + parameters['b' + str(i+1)]
        cache['A' + str(i+1)] = perform_activation(cache['Z' + str(i+1)], activations[i])

    cache['AL'] = cache['A' + str(L)]

    return cache


def perform_activation(Z, activation = 'sigmoid'):
    if activation == 'sigmoid':
        return sigmoid(Z)
    if activation == 'tanh':
        return np.tanh(Z)
    ##########


def activation_derivative(Z, activation = 'sigmoid'):
    s = perform_activation(Z, activation)

    if activation == 'sigmoid':
        return np.multiply(s, (1-s))
    if activation == 'tanh':
        return 1 - np.power(s, 2)


def compute_cost_default(Y, A):
    m = Y.shape[1]
    return (-1/m)*np.sum(np.multiply(Y, np.log(A)) + np.multiply((1-Y), np.log(1-A)))

def compute_cost_rgl_l2(Y, A, parameters, rgl_lambda=0):
    m = Y.shape[1]
    L = int(len(parameters) / 2)
    I_ce = (-1 / m) * np.sum(np.multiply(Y, np.log(A)) + np.multiply((1-Y), np.log(1-A)))
    I_rg = (1 / m) * (rgl_lambda / 2) * np.sum([np.sum(np.square(parameters['W' + str(l+1)])) for l in range(L)])
    return I_ce + I_rg

def cost_derivative(Y, A):
    return (1-Y)/(1-A) - (Y/A)


def backward_propagation_default(X, Y, parameters, cache, layers, activations):
    L = len(layers)
    m = X.shape[1]
    grads = dict()

    dAL = cost_derivative(Y, cache['A'+str(L)])
    dZL = np.multiply(activation_derivative(cache['Z'+str(L)], activation=activations[L-1]), dAL)
    dZ = dZL

    for l in range(L, 1, -1):
        grads['dW' + str(l)] = (1 / m) * np.dot(dZ, (cache['A' + str(l - 1)]).T)
        grads['db' + str(l)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dZ = np.multiply(np.dot(parameters['W' + str(l)].T, dZ), activation_derivative(cache['Z'+str(l-1)], activation=activations[l-2]))

    grads['dW1'] = (1 / m) * np.dot(dZ, X.T)
    grads['db1'] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

    return grads


def backward_propagation_rgl_l2(X, Y, parameters, cache, layers, activations, rgl_lambda=0):
    L = len(layers)
    m = X.shape[1]
    grads = dict()

    dAL = cost_derivative(Y, cache['A'+str(L)])
    dZL = np.multiply(activation_derivative(cache['Z'+str(L)], activation=activations[L-1]), dAL)
    dZ = dZL

    for l in range(L, 1, -1):
        grads['dW' + str(l)] = (1 / m) * np.dot(dZ, (cache['A' + str(l - 1)]).T) + (rgl_lambda/m)*parameters['W' + str(l)]
        grads['db' + str(l)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dZ = np.multiply(np.dot(parameters['W' + str(l)].T, dZ), activation_derivative(cache['Z'+str(l-1)], activation=activations[l-2]))

    grads['dW1'] = (1 / m) * np.dot(dZ, X.T) + (rgl_lambda/m)*parameters['W1']
    grads['db1'] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

    return grads




def update_parameters(parameters, grads, learning_rate):
    for l in range(int(len(parameters)/2)):
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - (learning_rate * grads['dW' + str(l + 1)])
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - (learning_rate * grads['db' + str(l + 1)])
    return parameters


def predict(X_orig, parameters, activations):
    #X = normalize(X_orig)
    X = X_orig
    #print(forward_propagation(X, parameters, activations)['AL'])
    return (forward_propagation(X, parameters, activations)['AL'] > 0.5).astype(int).flatten()


def model_dnn_amion(X_orig, Y_orig, layers = [1], activations = ['sigmoid'], num_iterations = 1000, learning_rate = 0.01, rgl_lambda = 0, init_way = 'default_rnd'):
    #X = normalize(X_orig)
    X = X_orig
    Y = Y_orig.reshape(1, Y_orig.size)

    if rgl_lambda == 0:
        compute_cost, backward_propagation = compute_cost_default, backward_propagation_default
    else:
        compute_cost, backward_propagation = compute_cost_rgl_l2, backward_propagation_rgl_l2

    parameters = initialize_parameters(X, layers, init_way)
    costs = []

    for i in (range(num_iterations)):

        cache = forward_propagation(X, parameters, activations)
        grads = backward_propagation(X, Y, parameters, cache, layers, activations)

        parameters = update_parameters(parameters, grads, learning_rate)

        if rgl_lambda == 0:
            I = compute_cost(Y, cache['AL'])
        else:
            I = compute_cost(Y, cache['AL'], parameters, rgl_lambda)
        costs.append(I)
        if i % 1000 == 0:
            print("Cost = ", I)

    #checking_gradient(X, Y, parameters, layers, activations, compute_cost, backward_propagation)

    print()


    plt.plot(costs)
    plt.ylabel('Cost-Function')
    plt.show()
    return parameters


def dict_to_tetta(parameters, layers, type = 'parameters'):
    new_param = parameters.copy()
    tetta = np.asarray([])

    if type == 'parameters':
        tp = ''
    elif type == 'grads':
        tp = 'd'
    else:
        raise AssertionError('Uncorrect type')

    for l in range(1, len(layers) + 1):
        tetta = np.concatenate((tetta, new_param[tp + 'W' + str(l)].flatten()))
        tetta = np.concatenate((tetta, new_param[tp + 'b' + str(l)].flatten()))
    return tetta


def tetta_to_dict(X, tetta, layers, type = 'parameters'):

    if type == 'parameters':
        tp = ''
    elif type == 'grads':
        tp = 'd'
    else:
        raise AssertionError('Uncorrect type')

    params = dict()
    layers_with_X = layers.copy()
    layers_with_X.insert(0, X.shape[0])
    m = 0

    for l in range(1, len(layers_with_X)):
        w_len = layers_with_X[l] * layers_with_X[l - 1]
        b_len = layers_with_X[l]
        params[tp + "W" + str(l)] = tetta[m: m + w_len].reshape(layers_with_X[l], layers_with_X[l - 1])
        params[tp + "b" + str(l)] = tetta[m + w_len: m + w_len + b_len].reshape(layers_with_X[l], 1)
        m += w_len + b_len

    return params


def checking_gradient(X, Y, parameters, layers, activations, compute_cost, backward_propagation):
    grads_test = dict()

    cache = forward_propagation(X, parameters, activations)
    grads = backward_propagation(X, Y, parameters, cache, layers, activations)

    da = 1e-7
    params = parameters.copy()

    tetta = dict_to_tetta(params, layers, type='parameters')
    dtetta = dict_to_tetta(grads, layers, type='grads')
    dtetta_approx = np.asarray([])

    for p in range(len(tetta)):

        tetta[p] += da
        params = tetta_to_dict(X, tetta, layers, type = 'parameters')
        I2 = compute_cost(Y, forward_propagation(X, params, activations)['AL'])

        tetta[p] -= 2*da
        params = tetta_to_dict(X, tetta, layers, type = 'parameters')
        I1 = compute_cost(Y, forward_propagation(X, params, activations)['AL'])

        dtetta_approx = np.append(dtetta_approx, (I2 - I1)/(2*da))
        tetta[p] += da

    check = np.linalg.norm(dtetta_approx-dtetta)/(np.linalg.norm(dtetta_approx) + np.linalg.norm(dtetta))
    print("gradient_check = ", check)
    return check
