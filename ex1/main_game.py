##
# Main function of the Python program.
#
##

# from histogram_mine import *
from histogram_game import *


def main():
    # np.random.seed(42)
    Map = np.random.randint(0, 2, (100, 100)).astype(str)
    plt.matshow(Map.astype(int))
    plt.figure()
    Map[Map == '0'] = 'G'
    Map[Map == '1'] = 'R'

    sensor_right = .8
    p_move = .8
    ans = histogram_localization(Map, sensor_right, p_move)
    show(ans)  # displays your answer
    print(ans[0])



if __name__ == '__main__':
    main()
