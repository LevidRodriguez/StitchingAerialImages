import java.util.Scanner;
 
public class Pila {
 
    Scanner teclado = new Scanner(System.in);
    int pila[]=new int[5];
    int tope=-1;
 
    public int push(){
        if(tope>=pila.length-1){
            System.out.println("la pila esta llena");
        }
        else{
            tope+=1;
            System.out.println("ingrese el dato");
            pila[tope]=teclado.nextInt();
        }
        return tope;
    }
 
    public int pop(){
        if(tope==-1){
            System.out.println("La pila esta vacia");
        }
        else{
            System.out.println("Se elimino un elemento de la pila");
            pila[tope]=0;
            tope-=1;
        }
        return tope;
    }
 
    public void printPila(){
        for(int tope=4;tope>=0;tope--){
            System.out.println("Datos de la pila: "+pila[tope]);
        }
    }
 
    public static void main(String[] args) {
        Pila platos=new Pila();
        platos.push();
        platos.push();
        platos.printPila();
        platos.pop();
        platos.pop();
        platos.pop();

    }
}