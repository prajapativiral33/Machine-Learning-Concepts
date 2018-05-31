from numpy import *

def compute_error_for_line_given_points(b,m,points):
    #initialize the error with zero
    error = 0
    for i in range(0,len(points)):
        #take x value
        x = points[i,0]

        #take y value
        y = points[i,1]

        #ecuate the formula    (1/N)* i(0,N) E(yi - (m * xi + b))^2
        error += (y - (m * x + b)) ** 2
    return error/len(points)


def gradient_decent(points,b,m,learning_rate,num_of_iterations):

    for i in range(num_of_iterations):
        b, m = gradient_decent_steps(b,m,array(points),learning_rate)
    return [b,m]

def gradient_decent_steps(b_current,m_current,points,learning_rate):

    #Gradient decent m & b formulas
    # derivatives of m =   (-2/N) i(1,N)E  (xi)( yi - (m_current * xi + b_current))
    # derivatives of b =   (-2/N) i(1,N)E ( yi - (m_current * xi + b_current))
    m_gradient = 0
    b_gradient = 0
    N = float(len(points))
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]

        m_gradient +=  (-2/N)*(x * (y - (m_current * x + b_current)))
        b_gradient +=  (2/N)*(y - (m_current * x + b_current))

    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [ new_b,new_m]

def run():

    #Step-1   : Load the data /Collect our data
    points = genfromtxt('data.csv', delimiter = ',')

    #Step -2 : Define the hyperparameters

    learning_rate = 0.0001   #How fast our model will be
    #initialize b & m : y = mx + b
    initial_b = 0
    initial_m = 0
    num_of_iterations = 1000

    #Step -3 : tarin the model
    print 'starting gradient decent at b={0},m={1},error ={2}'.format(initial_b,initial_m,compute_error_for_line_given_points(initial_b,initial_m,points))
    [b,m] = gradient_decent(points,initial_b,initial_m, learning_rate,num_of_iterations)
    print 'ending gradient decent at b={1},m={2},error={3}'.format(num_of_iterations,b,m,compute_error_for_line_given_points(b,m,points))


if __name__ == '__main__':
    run();
