#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "my_utils.h"
#define byte unsigned char

byte *d_sbox;
byte *d_m2;
byte *d_m3;
byte *d_rcon;

long time_counter = 0; //Para contar el tiempo de ejecucion
long time_total, time_encry, time_partial;


/****************************** CPU *************************************/

//Leer el archivo de llave de la SD y guardarla en la variable global de key[]
void read_key_from_file(byte *key) {
   FILE *fp;
   byte *key_file = "key.txt";
   byte buffer[20];//Buffer para almacenar el bloque de datos leidos
   byte i;

   if((fp = fopen(key_file, "rb")) == NULL){
       printf(" Cannot open input file: %s\n",key_file);
       printf(" Exiting program...\n");
       return;
   }

   fgets(buffer, 16, (FILE*)fp);
   for(i = 0;i < 16;i++) {
      key[i]=buffer[i];
   }                                                                                                                
   fclose(fp);
   printf("Key stored correctly.\n");
}


unsigned long get_file_size(FILE *f){
    int prev = ftell(f);
    fseek(f, 0L, SEEK_END);
    int size = ftell(f);
    fseek(f, prev, SEEK_SET);
    return size;
}

// Auxiliar functions 
void print_state(byte *state) {
   for(int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
         printf("%0X ", state[i*4+j]);
      }
      printf("\n");
   }
}

/*Obtiene el valor correspondiente para el byte de la tabla S-BOX (Para cifrado)*/
byte get_sbox(byte pos) {return d_sbox[pos];}
byte mul_2(byte a) { return d_m2[a]; }
byte mul_3(byte a) { return d_m3[a]; }

/*Realiza una rotacion circular de un byte a la izquierda*/
void rotateLeft(byte *A) {    
    byte i;
    byte aux = A[0];    
    for(i=0;i<3;i++) {
        A[i] = A[i+1];
    }
    A[3] = aux;
}
/*************************************** GPU *************************************/

/*Funcion que mezcla la llave expandida con el bloque de estado*/
void add_round_key(byte *state, int round, byte *expanded_key) {
    /*El bloque de la llave expandida depende del numero de round*/
    int i, j;
    for(i=0;i<4;i++) {
        for(j=0;j<4;j++) {
            state[i * 4 + j] ^= expanded_key[round*16 + i*4 + j];
        }        
    }
}

void subbytes(byte *state) {
    byte i, j;
    for(i=0;i<4;i++) 
        for(j=0;j<4;j++) 
            state[j * 4 + i] = get_sbox(state[j * 4 + i]);
}


void shift_rows(byte *state) {    
    byte i;
    byte *temp = (byte*)malloc(4);

    memcpy(temp, state + 4, 4);
    rotateLeft(temp);        
    memcpy(state + 4, temp, 4);

    memcpy(temp, state + 8, 4);
    rotateLeft(temp);
    rotateLeft(temp);
    memcpy(state + 8, temp, 4);

    memcpy(temp, state + 12, 4);
    rotateLeft(temp);
    rotateLeft(temp);
    rotateLeft(temp);
    memcpy(state + 12, temp, 4);

    free(temp);
}



/*Substitución que usa aritmética de campos finitos sobre GF(2^^8).*/
void mix_columns(byte *state) {
    byte i, a0, a1, a2, a3;
    for(i=0;i<4;i++) {
        a0 = state[i * 4 + 0];
        a1 = state[i * 4 + 1];
        a2 = state[i * 4 + 2];
        a3 = state[i * 4 + 3];

        state[i * 4 + 0] = mul_2(a0) ^ mul_3(a1) ^ a2 ^ a3;
        state[i * 4 + 1] = mul_2(a1) ^ mul_3(a2) ^ a0 ^ a3;
        state[i * 4 + 2] = mul_2(a2) ^ mul_3(a3) ^ a0 ^ a1;
        state[i * 4 + 3] = mul_2(a3) ^ mul_3(a0) ^ a1 ^ a2;        
    }
}

void cipher(byte *state, byte *expanded_key) {    
    int round=0;    
    add_round_key(state, round, expanded_key);    
    for(round=1; round < 10 ; round++) {
        subbytes(state);
        shift_rows(state);
        mix_columns(state);
        add_round_key(state, round, expanded_key);
    }
    subbytes(state);
    shift_rows(state);
    add_round_key(state, 10, expanded_key);    
}

// To expand the key using the sbox matrix
void key_expansion(byte *key, byte *expanded_key) {
    byte temp[4];
    byte c=16;
    byte j, a, i=1;
    for(j=0; j < 16; j++) {
        expanded_key[j] = key[j];
    }
    while(c < 176) {
        for(a = 0; a < 4; a++) {
            temp[a] = expanded_key[a+c-4];
        }
        if(c % 16 == 0) {
            rotateLeft((byte*)&temp);
            for(a = 0; a < 4; a++) {
                temp[a] = get_sbox(temp[a]);
            }
            temp[0] =  temp[0] ^ d_rcon[i];
            i++;
        }
        for(a = 0; a < 4; a++) {
            expanded_key[c] = expanded_key[c-16] ^ temp[a];
            c++;
        }
    }
}

// To read a file from the PC and to copy the binary data to an array of bytes
// returns the size of the file or -1 if there was an error
long long load_file(byte *file_in_name, byte **file_in) {
    FILE *f_i; // Pointer to the file
    unsigned long size; // Size of the file

    // Open file
    if((f_i = fopen((const char*)file_in_name,"rb")) == NULL)  {
        printf("Error trying to open the file %s\n", file_in_name);
        return -1;
    }          
    // Compute size of the file          
    size = get_file_size(f_i);

    // Allocate memory to store the input file as binary data
    *file_in = (byte*)malloc(size * sizeof(byte));
    // Read all the data from the file and save it in the array
    fread(*file_in, sizeof(byte), size, f_i);

    // Close file (no longer required)
    fclose(f_i);

    return size;
}

void write_file(byte *file_out_name, byte *file_out, long long *file_out_size) {
    FILE *f_o; // Pointer to the file

    // Open file
    if((f_o = fopen((const char*)file_out_name,"wb")) == NULL)  {
        printf("Error trying to open the file %s\n", file_out_name);
        return;
    }

    fwrite(file_out, sizeof(byte), *file_out_size, f_o);

    fclose(f_o);
}


void cipher_control(byte *file_in_name, byte *file_in, byte *file_out, long long *file_size, unsigned long *blocks, byte *expanded_key) {
    byte state[16];
    unsigned long block;
                 
    for(block = 1; block <= *blocks; block++) {
        memcpy(state, file_in + (block - 1) * 16, 16 * sizeof(byte));
        //fread(buffer, 16, 1, file_in); //leer archivo
        //Copiar bloque al state           
        for(int i = 0;i < 4;i++) {  
            for(int j = 0;j < 4;j++) {
                //state[i * 4 + j] = buffer[ i * 4 + j];
                //Padding para el cifrado
                if(i * 4 + j >= *file_size) state[i * 4 + j] = 0x00;//Esto solo sucedera en el cifrado, no es necesario condicionar
            }                        
        }     

        cipher(state, expanded_key);

        //Copiar el state a un nuevo bloque para escribirlo en el archivo encriptado                
        /*for(int i = 0;i < 4;i++) {  
            for(int j = 0;j < 4;j++) {
                buffer[i * 4 + j] = state[i * 4 + j];
            }                        
        }
        */

        // Copy the encrypted block to the output file
        memcpy(file_out + (block - 1) * 16, state, 16 * sizeof(byte));
         
        /*//Informar estado
        if( (blocks < 500) || (blocks >= 500 && block%10==0) || (block==blocks)) {
            printf("Trabajando... [%lu/%lu]\n\r", block, blocks);
        } 
        */               
    }
}





int main() {
    // Pointer to data in the HOST memory
    byte *file_in; // Stores the binary data of the file to be encrypted
    byte *file_out; // Stores the binary data of the encrypted file
    byte *key; // Stores the key provided by the user 
    byte *expanded_key; // The key after the expansion algorithm
    long long *file_in_size, *file_out_size;
    unsigned long *blocks, padding; // Number of blocks to divide the input file for the AES process
    // GPU variables
    /*byte *d_sbox;
    byte *d_m2;
    byte *d_m3;
    byte *d_rcon;
*/

    // Name of the input and output files
    byte *file_name = "files/test.jpg";
    byte *out_file_name = "files/test.aes";

    int byte_size = sizeof(byte);
    // Allocate HOST memory
    key = (byte*)malloc(16 * byte_size);
    expanded_key = (byte*)malloc(176 * byte_size);
    //state = (byte*)malloc(16 * byte_size);
    blocks = (unsigned long*)malloc(sizeof(unsigned long));
    file_in_size = (long long*)malloc(sizeof(long long));
    file_out_size = (long long*)malloc(sizeof(long long));

    // Allocate DEVICE memory
    //cudaMalloc((void**)&d_sbox, byte_size * 256);
    d_sbox = (byte*) malloc(byte_size * 256);
    d_m2 = (byte*) malloc(byte_size * 256);
    d_m3 = (byte*) malloc(byte_size * 256);
    d_rcon = (byte*) malloc(byte_size * 11);


    memcpy(d_sbox, SBOX, byte_size * 256);
    memcpy(d_m2, M2, byte_size * 256);
    memcpy(d_m3, M3, byte_size * 256);
    memcpy(d_rcon, RCON, byte_size * 11);
       
    /* Starting encryption pre-process */

    // Load the file to be encrypted
    *file_in_size = load_file(file_name, &file_in);
    
    // Compute the number of blocks needed and check wheter the file requires a byte of
    // padding at the end (when it is not a multiple of 16)
    *blocks = (*file_in_size) / 16;
    padding = 0;
    if(*file_in_size % 16 != 0) {
        padding = 16 - (*file_in_size) % 16;
        (*blocks)++;
    }
    // The size of the output file will be a multiple of 16 + 1 because of the byte at the beggining to indicate the padding
    *file_out_size = (*blocks) * 16 + 1;

    // Allocate the memory for the output file
    file_out = (byte*)malloc((*file_out_size) * byte_size);
    // REMEMBER !!!!!!!!!!!!!!!!!!!! // Write in the first byte of the output the number of padding bytes
    file_out[0] = padding;
     

    // Read the key used to encrypt
    read_key_from_file(key);
    key_expansion(key, expanded_key);


    // GPU memory allocation
    // GPU memory copy
    //cudaMemcpy(d_sbox, SBOX, byte_size * 256, cudaMemcpyHostToDevice);

    printf("Starting encryption...\n");
    cipher_control(file_name, file_in, file_out + 1, file_in_size, blocks, expanded_key);
    printf("Done encryption.\n");

    write_file(out_file_name, file_out, file_out_size);

    
    
    free(file_in);
    free(file_out);
    free(file_in_size);
    free(file_out_size);
    free(blocks);
    free(key);
    free(expanded_key);
    //free(state);
    
    return 0; 
}