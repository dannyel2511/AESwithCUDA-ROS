#include "ros/ros.h"
#include <time.h>
#include "std_msgs/Int32.h"
#include "std_msgs/String.h"
#include "sensor_msgs/Image.h"

#define byte unsigned char

#define N (1024*1024)

#define THREADS_PER_BLOCK 512

byte* *decipher_main(byte *data, int data_size, byte **ans, byte *cip_key);
int size=10;

bool ready = false;

std_msgs::String cip_msg;
std_msgs::String decip_msg;

void chatterCallback(const std_msgs::String::ConstPtr& msg)
{
    if(!ready) {
        cip_msg.data = msg->data;
        ready = true;
    }
  //ROS_INFO("I heard: [%s] . CPP", msg->data.c_str());
}

void print_state_cpu(byte *state) {
   for(int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
         printf("%0X ", state[i*4+j]);
      }
      printf("\n");
   }
   printf("\n");
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "roscuda_aes_decipher");	  
	ros::NodeHandle n;
	
	ros::Subscriber sub = n.subscribe("ciphered_msg", 1000, chatterCallback);
	
	byte *data = (byte*)malloc(100 * sizeof(byte));
	byte *deciphered_data = (byte*)malloc(100 * sizeof(byte));
	byte my_key[16] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F};
	

	while(ros::ok())
	{    

        if(ready) {
		    //data = cip_msg.data;
            //memcpy(data, cip_msg.data, sizeof(cip_msg.data)/sizeof(cip_msg.data[0]));

		    printf("Initial\n");
		    print_state_cpu(data);

		    decipher_main(data, sizeof(data)/sizeof(data[0]), &deciphered_data, my_key);		
		
		    printf("Deciphered\n");
		    print_state_cpu(deciphered_data);

            decip_msg.data = (char*)deciphered_data;
	
            ROS_INFO("MSG: [%s] ", decip_msg.data.c_str());
            ready = false;
        }


	    ros::spinOnce();
	}
	return 0;
}
