##
# Main function of the Python program.
#
##

from histogram import *

def main():

    Map = [['G', 'G', 'G'],
          ['G', 'R', 'R'],
          ['G', 'G', 'G']]
    measurements = ['R','R']
    motions = [[0,0],[0,1]]
    sensor_right = .8
    p_move = 1.0
    ans = histogram_localization(Map, measurements, motions, sensor_right, p_move)
    show(ans) # displays your answer
    print(ans[0])


if __name__ == '__main__':
    main()
