import re

text = '''
# ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱ Assignment 1.1 ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
# TODO:                                                             #
# Calculate the L2 distance between the ith test point and the jth  #
# training point and store the result in dists[i, j]. Avoid using   #
# loops over dimensions or np.linalg.norm().                        #
# ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
# 🌀 INCEPTION 🌀 (Your code begins its journey here. 🚀 Do not delete this line.)

dif_vector = X[i] - self.X_train[j]
distance = np.sqrt(np.dot(dif_vector, dif_vector))
dists[i, j] = distance

# 🌀 TERMINATION 🌀 (Your code reaches its end. 🏁 Do not delete this line.)
'''

replacement = '''
#                    ╔═══════════════════════╗
#                    ║                       ║
#                    ║       YOUR CODE       ║
#                    ║                       ║
#                    ╚═══════════════════════╝
'''

regex = r'(?<=🌀 INCEPTION 🌀 \(Your code begins its journey here. 🚀 Do not delete this line.\)\n)([\s\S]*?)(?=\n\s*# 🌀 TERMINATION 🌀 \(Your code reaches its end. 🏁 Do not delete this line.\))'
result = re.sub(regex, replacement, text)

print(result)
