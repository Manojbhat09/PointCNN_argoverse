import rosbag
import sys 

num_msgs = 100
arguments = len(sys.argv) - 1

inputbag = sys.argv[1]
outputbag = sys.argv[2]
print(inputbag,  " ", outputbag)


with rosbag.Bag(outputbag, 'w') as outbag:
    for topic, msg, t in rosbag.Bag(inputbag).read_messages():
        # This also replaces tf timestamps under the assumption 
        # that all transforms in the message share the same timestamp
        if topic == "/tf" and msg.transforms:
            outbag.write(topic, msg, msg.transforms[0].header.stamp)
        else:
            outbag.write(topic, msg, msg.header.stamp if msg._has_header else t)
        num_msgs -= 1
        if num_msgs == 0:
            break
