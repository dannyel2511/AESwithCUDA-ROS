#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "my_utils.h"

typedef unsigned char byte;

// Variables to open a file
FILE *fp;

/*Variables globales para el proceso de AES*/
byte *state; /*Variable global que contiene el estado con el que se esta trabajando actualmente en el AES*/
// Stores the key provided by the user
byte *key;
// The key after the expansion algorithm
byte *expanded_key;/*Guarda la llave expandida*/

unsigned int br;
long time_counter=0; //Para contar el tiempo de ejecucion
long time_total, time_encry, time_partial;

/*Obtiene el valor correspondiente para el byte de la tabla S-BOX (Para cifrado)*/
byte getSBox(byte pos) {return sbox[pos];}

/*Realiza una rotacion circular de un byte a la izquierda*/
void rotateLeft(byte *A) {    
    byte i;
    byte aux = A[0];    
    for(i=0;i<3;i++) {
        A[i] = A[i+1];
    }
    A[3] = aux;
}

/*Se utiliza para generar una subllave para cada ronda a partir de la llave original.*/
void keyExpansion() {
    byte temp[4];
    byte c=16;
    byte j, a, i=1;
    for(j=0; j < 16; j++) {
        expanded_key[j] = key[j];
    }
    while(c<176) {
        for(a = 0; a < 4; a++) {
            temp[a] = expanded_key[a+c-4];
        }
        if(c % 16 == 0) {
            rotateLeft((byte*)&temp);
            for(a = 0; a < 4; a++) {
                temp[a] = getSBox(temp[a]);
            }
            temp[0] =  temp[0] ^ rcon[i];
            i++;
        }
        for(a = 0; a < 4; a++) {
            expanded_key[c] = expanded_key[c-16] ^ temp[a];
            c++;
        }
    }
}
/*Funcion que mezcla la llave expandida con el bloque de estado*/
void addRoundKey(int round) {
    /*El bloque de la llave expandida depende del numero de round*/
    int i, j;
    for(i=0;i<4;i++) {
        for(j=0;j<4;j++) {
            state[i * 4 + j] ^= expanded_key[round*16 + i*4 + j];
        }        
    }
}
/*Utiliza una matriz S-box para realizar una substitución byte a byte del bloque del estado.*/
void subBytes() {
    byte i, j;
    for(i=0;i<4;i++) 
        for(j=0;j<4;j++) 
            state[j * 4 + i] = getSBox(state[j * 4 + i]);
}

/*Realiza una permutación simple de bytes*/
/*    1er renglón sin cambios*/
/*    2o renglón con rotación circular de 1 bytes a la izquierda.*/
/*    3er renglón con rotación circular de 2 bytes a la izquierda.*/
/*    4o renglón con rotación circular de 3 bytes a la izquierda.*/
void shiftRows() {    
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
/*Funciones auxiliares para el MixColumns*/
byte mul_2(byte a) { return M2[a]; }
byte mul_3(byte a) { return M3[a]; }


/*Substitución que usa aritmética de campos finitos sobre GF(2^^8).*/
void mixColumns() {
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

void print_state() {
   for(int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
         printf("%0X ", state[i*4+j]);
      }
      printf("\n");
   }
}

void cipher() {    
    int round=0;    
    addRoundKey(round);    
    for(round=1; round < 10 ; round++) {
        //printf("----------Ronda %d------------\n", round);        
        //print_state();
        subBytes();
        //printf("Despues de SubBytes\n");
        //print_state();
        shiftRows();
        //printf("Despues de shiftRows\n");
        //print_state();
        mixColumns();
        //printf("Despues de mixColumns\n");
        //print_state();
        addRoundKey(round);        
    }
    subBytes();
    shiftRows();    
    addRoundKey(10);    
}


//Leer el archivo de llave de la SD y guardarla en la variable global de key[]
void read_key_from_file() {
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
   printf("Key stored correctly.");
}

byte hex_value(byte x) {    
    if(x>='0' && x <= '9') return x-'0';
    if(x>='a') return x-'a'+10; 
    return x-'A'+10;
}

int get_file_size(FILE *f){
    int prev = ftell(f);
    fseek(f, 0L, SEEK_END);
    int size = ftell(fp);
    fseek(f, prev, SEEK_SET);
    return size;
}

void cipher_control(byte *f_i) {
    byte file_route[100];//Ruta para buscar el archivo
    FILE *file_in;//Archivo de lectura
    FILE *file_out;//Archivo de escritura

    byte *my_input;
    byte buffer[20];

    int i,j;
    unsigned long size, block, blocks, padding;

    
    printf("Comienza lectura de archivos y cifrado.\n\r");

    sprintf(file_route, "files/%s", f_i);//Ruta completa del archivo para cifrar

    //Abrir archivo de entrada  (lectura)      
    if((file_in = fopen(file_route,"rb")) == NULL)  {
        printf("El archivo [%s] no se pudo abrir\n\r", f_i);
        return;
    }          
    //Calcular tamanio y cantidad de bloques                 
    size = get_file_size(file_in);
    // Allocate memory to store the input file
    my_input = (byte*)malloc(size * sizeof(byte));
    fread(my_input, sizeof(byte), size, file_in); //leer archivo

    blocks = size / 16;

    if(size % 16 != 0)//Solo aplica para cifrar, ya que para descifrar los bloques siempre son exactos
        blocks++; //Hay un bloque que no llega a 16 bytes
     
    sprintf(file_route, "files/%s.aes", f_i);//Ruta completa del archivo a cifrar
    if((file_out = fopen(file_route, "wb")) == NULL) {
        printf("Error abriendo el archivo para cifrar\n");
        return;
    }

    printf("Cifrando...\n");
    //Escribir cuantos bytes de padding se agregaran
    //Eso es en el primer byte, se escribe al archivo de salida, luego se ignora y comienza el proceso normal
    buffer[0] = 0;
    if(size % 16 != 0) buffer[0] = 16 - size % 16;                                                                                                                          
    fwrite(buffer, sizeof(byte), 1, file_out);


    printf("Trabajando...\n");
    //Comenzar a contar tiempo
    time_counter = time_total = time_encry = 0;
    //Poner algo para contar                  
                 
    for(block = 1; block <= blocks; block++) {
        memcpy(buffer, my_input + (block - 1) * 16, 16 * sizeof(byte));
        //fread(buffer, 16, 1, file_in); //leer archivo
        //Copiar bloque al state           
        for(i=0;i<4;i++) {  
            for(j=0;j<4;j++) {
                state[i * 4 + j]=buffer[ i * 4 + j];
                //Padding para el cifrado
                if(i*4+j >= size) state[i * 4 + j] = 0x00;//Esto solo sucedera en el cifrado, no es necesario condicionar
            }                        
        }                
        time_partial = time_counter;      
        cipher(); //Cifrar el bloque
        time_encry += (time_counter - time_partial);
                             
        //Copiar el state a un nuevo bloque para escribirlo en el archivo encriptado                
        for(i=0;i<4;i++) {  
            for(j=0;j<4;j++) {
                buffer[i * 4 + j] = state[i * 4 + j];
            }                        
        }
        fwrite(buffer, sizeof(byte), 16, file_out); //Escribir el bloque cifrado al archivo
         
        /*//Informar estado
        if( (blocks < 500) || (blocks >= 500 && block%10==0) || (block==blocks)) {
            printf("Trabajando... [%lu/%lu]\n\r", block, blocks);
        } 
        */               
    }                                                                          
    fclose(file_in);           
    fclose(file_out);
    free(my_input);
     
    //Apagar el timer            
     
    //Informar estado.            
     
    printf("Proceso terminado con exito.\n\r");            
    printf("Entrada: %s\n\r", f_i);
    printf("Numero de bytes cifrados: [%lu]\n\r", size);
    printf("Bloques de 128 bits: [%lu]\n\r",blocks);
    printf("Tiempo total: %lu.%lu s\n\r", time_counter/10, time_counter%10);
    printf("Tiempo del cifrado: %lu.%lu s\n\r", time_encry/10, time_encry%10);           
    printf("Tiepo promedio por bloque %.5fs\n\r", (float)time_encry / blocks);                                          

}



int main() {
    byte *file_name; //Cadena auxiliar para guardar el nombre de archivo de entrada
    byte *out_file_name; //Cadena auxiliar para guardar el nombre de archivo de salida    
    byte input; //Opcion que seleeciona el usuario
    int byte_size = sizeof(byte);
    // Allocate HOST memory
    key = (byte*)malloc(16 * byte_size);
    expanded_key = (byte*)malloc(176 * byte_size);
    state = (byte*)malloc(16 * byte_size);
                                              
    /*------------------------Interfaz de usuario--------------------------*/
       
    printf("Cipher process started\n\r");
   
    file_name = "test.jpg";
    out_file_name = "test";
    read_key_from_file();//Solicitar que se introduzca la llave  
    keyExpansion();
    //Funcion para comenzar cifrado del archivo
    cipher_control(file_name);
    // cipher_or_decipher_file(file_name, out_file_name,1);                                               
    
    free(key);
    free(expanded_key);
    free(state);
    return 0;
      
}