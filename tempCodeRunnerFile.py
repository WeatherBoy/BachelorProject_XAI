    temp1, temp2 = 0, 0
            if lower_quantiles[idx] < unique_occurence:
                temp1 = np.absolute(unique_occurence - lower_quantiles[idx])
            if unique_occurence < upper_quantiles[idx]:
                temp2 = np.absolute(unique_occurence - upper_quantiles[idx])
            
            x_vals = [x_positions[idx] + deltas[indx], x_positions[idx] + deltas[indx]]
            plt.plot(x_vals, [unique_occurence + temp2, unique_occurence - temp1], color = COLOURS[indx], linewidth = 1.2)