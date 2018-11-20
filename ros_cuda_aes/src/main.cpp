#include "ros/ros.h"
#include <time.h>
#include "std_msgs/Int32.h"
#include "std_msgs/String.h"
#include "sensor_msgs/Image.h"

#define byte unsigned char

#define N (1024*1024)

#define THREADS_PER_BLOCK 512

byte* *cipher_main(byte *data, int data_size, byte **ans, byte *cip_key);
int size=10;



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
	ros::init(argc, argv, "roscuda_aes_cipher");	  
	ros::NodeHandle n;
	
	ros::Publisher pub = n.advertise<std_msgs::String>("/ciphered_msg", 1);
	std_msgs::String msg;
	//image_transport::ImageTransport it(n);
   //image_transport::Subscriber sub = it.subscribe("image_raw", 1, imageCallback);
   //image_transport::Publisher pub = it.advertise("dannyel_image", 1);
	
	byte *data = (byte*)malloc(100 * sizeof(byte));
	byte *ciphered_data = (byte*)malloc(100 * sizeof(byte));
	byte my_key[16] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F};
	

	while(ros::ok())
	{    

		data = (byte*)"Este es un mensaje cifrado";

		printf("Inicial\n");
		print_state_cpu(data);

		cipher_main(data, sizeof(data)/sizeof(data[0]), &ciphered_data, my_key);		
		//memcpy(cv_frame.data, ciphered_data, sizeof(cv_frame.data)/sizeof(cv_frame.data[0]));
		
		printf("Cifrado\n");
		print_state_cpu(ciphered_data);
	
		msg.data = (char*)ciphered_data;
		//msg.data = (char*)data;

		pub.publish(msg);


	    ros::spinOnce();
	}
	return 0;
}
