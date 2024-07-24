import re

# Define the different types of tokens
NUMBER = 'NUMBER'
PLUS = 'PLUS'
MINUS = 'MINUS'
TIMES = 'TIMES'
DIVIDE = 'DIVIDE'
LPAREN = 'LPAREN'
RPAREN = 'RPAREN'

# Token specifications
token_specification = [
    (NUMBER,   r'\d+(\.\d*)?'),  # Integer or decimal number
    (PLUS,     r'\+'),           # Addition operator
    (MINUS,    r'-'),            # Subtraction operator
    (TIMES,    r'\*'),           # Multiplication operator
    (DIVIDE,   r'/'),            # Division operator
    (LPAREN,   r'\('),           # Left parenthesis
    (RPAREN,   r'\)'),           # Right parenthesis
    ('SKIP',   r'[ \t]'),        # Skip spaces and tabs
    ('MISMATCH', r'.'),          # Any other character
]

# Regular expression for the tokenizer
token_re = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)


def tokenize(code):
    tokens = []
    for mo in re.finditer(token_re, code):
        kind = mo.lastgroup
        value = mo.group()
        if kind == NUMBER:
            value = float(value) if '.' in value else int(value)
        elif kind == 'SKIP':
            continue
        elif kind == 'MISMATCH':
            raise RuntimeError(f'{value!r} unexpected')
        tokens.append((kind, value))
    return tokens


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.current_token = None
        self.next_token()

    def next_token(self):
        self.current_token = self.tokens.pop(0) if self.tokens else None

    def parse(self):
        return self.expr()

    def eat(self, token_type):
        if self.current_token and self.current_token[0] == token_type:
            self.next_token()
        else:
            raise RuntimeError(f'Expected {token_type} but got {self.current_token}')

    def factor(self):
        token = self.current_token
        if token[0] == NUMBER:
            self.eat(NUMBER)
            return token[1]
        elif token[0] == LPAREN:
            self.eat(LPAREN)
            result = self.expr()
            self.eat(RPAREN)
            return result
        elif token[0] == MINUS:
            self.eat(MINUS)
            return -self.factor()

    def term(self):
        result = self.factor()
        while self.current_token and self.current_token[0] in (TIMES, DIVIDE):
            token = self.current_token
            if token[0] == TIMES:
                self.eat(TIMES)
                result *= self.factor()
            elif token[0] == DIVIDE:
                self.eat(DIVIDE)
                result /= self.factor()
        return result

    def expr(self):
        result = self.term()
        while self.current_token and self.current_token[0] in (PLUS, MINUS):
            token = self.current_token
            if token[0] == PLUS:
                self.eat(PLUS)
                result += self.term()
            elif token[0] == MINUS:
                self.eat(MINUS)
                result -= self.term()
        return result


def get_result(expression):
    tokens = tokenize(expression)
    parser = Parser(tokens)
    return parser.parse()


# # Usage example:
# expression = "(400 + (80 / 2) + 450 - (50 / 2)) / 2"
# tokens = tokenize(expression)
# print(tokens)
# parser = Parser(tokens)
# result = parser.parse()
# print(result)  # Output: 33
# print((400 + (80 / 2) + 450 - (50 / 2)) / 2)
