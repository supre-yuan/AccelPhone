from matplotlib import pyplot as plt


def showcutting(cuttingpoints, t_x_list_filter_two, x_list_filter_two, y_list_filter_two, z_list_filter_two, left,
                right, path):
    cuttingpoints_two = cuttingpoints[:]
    try:
        for i in range(len(cuttingpoints_two)):
            if (i % 2 == 0):
                if (cuttingpoints_two[i] - left <= 0):
                    cuttingpoints_two[i] = 0
                else:
                    cuttingpoints_two[i] = cuttingpoints_two[i] - left
                if (cuttingpoints_two[i + 1] + right > len(t_x_list_filter_two) - 1):
                    cuttingpoints_two[i + 1] = len(t_x_list_filter_two) - 1
                else:
                    cuttingpoints_two[i + 1] = cuttingpoints_two[i + 1] + right
    except:
        pass

    plt.plot(t_x_list_filter_two[:], z_list_filter_two[:], t_x_list_filter_two[cuttingpoints_two],
             z_list_filter_two[cuttingpoints_two], '*', markersize=20, lw=1)
    plt.title('Filtered signal with cutting points (z-axis)', fontsize=30)
    plt.xlabel('Time (secs)', fontsize=30)
    plt.ylabel('Acc (m/s\N{SUPERSCRIPT TWO})', fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(path, bbox_inches='tight')
    plt.show()


def showcutting_save(filename, cuttingpoints, t_x_list_filter_two, x_list_filter_two, y_list_filter_two,
                     z_list_filter_two, left,
                     right):
    cuttingpoints_two = cuttingpoints[:]
    for i in range(len(cuttingpoints_two)):
        if (i % 2 == 0):
            cuttingpoints_two[i] = cuttingpoints_two[i] - left
            cuttingpoints_two[i + 1] = cuttingpoints_two[i + 1] + right

    plt.subplot(3, 1, 1)
    plt.plot(t_x_list_filter_two[:], x_list_filter_two[:], t_x_list_filter_two[cuttingpoints_two],
             x_list_filter_two[cuttingpoints_two], '*')
    plt.title('Cutting in the Interp Map_X')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.subplot(3, 1, 2)
    plt.plot(t_x_list_filter_two[:], y_list_filter_two[:], t_x_list_filter_two[cuttingpoints_two],
             y_list_filter_two[cuttingpoints_two], '*')
    plt.title('Cutting in the Interp Map_Y')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.subplot(3, 1, 3)
    plt.plot(t_x_list_filter_two[:], z_list_filter_two[:], t_x_list_filter_two[cuttingpoints_two],
             z_list_filter_two[cuttingpoints_two], '*')
    plt.title('Cutting in the Interp Map_Z')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(filename + ".png")
    plt.show()
