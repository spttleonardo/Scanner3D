// Programa: Ponte H L298N com controle serial para scanner 3D
// Motor de passo controlado via comando 'R'
// Sinaliza com 'C' após cada giro e 'Q' ao completar 360º

const int IN1 = 8;
const int IN2 = 9;
const int IN3 = 10;
const int IN4 = 11;

const int tempo = 10;  // ms entre passos
const int STEPS_PER_CLICK = 1;  // passos por comando R

// Quantidade total de passos para completar 360º (ajuste conforme seu motor / redução)
const int TOTAL_STEPS_360 = 200;  

int currentStep = 0;
int totalSteps = 0;

void setup() {
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  Serial.begin(9600);
  // Serial.println("Scanner pronto. Envie 'R' para girar.");
}

void passo(int seq) {
  switch(seq) {
    case 0:
      digitalWrite(IN1, HIGH);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, HIGH);
      break;
    case 1:
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, HIGH);
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, HIGH);
      break;
    case 2:
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, HIGH);
      digitalWrite(IN3, HIGH);
      digitalWrite(IN4, LOW);
      break;
    case 3:
      digitalWrite(IN1, HIGH);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, HIGH);
      digitalWrite(IN4, LOW);
      break;
  }
  delay(tempo);
}

void desligarMotor() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}

void loop() {
  if (Serial.available() > 0) {
    char c = Serial.read();

    if (c == 'R' || c == 'r') {
      for (int i = 0; i < STEPS_PER_CLICK; i++) {
        passo(currentStep);
        currentStep = (currentStep + 1) % 4;
        totalSteps++;
      }

      desligarMotor();

      if (totalSteps >= TOTAL_STEPS_360) {
        Serial.println("Q");  // sinaliza final da rotação completa
        totalSteps = 0;       // reseta para nova rotação se quiser continuar
      } else {
        Serial.println("C");  // sinaliza pronto para próxima captura
      }

      // limpa buffer residual
      while (Serial.available() > 0) {
        Serial.read();
      }
    }
  }
}
