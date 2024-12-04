import re

# Define token types
TOKEN_TYPES = {
    'NUMBER': r'\d+(\.\d*)?',    # Integer or decimal number
    'PLUS': r'\+',               # Addition operator
    'MINUS': r'-',               # Subtraction operator
    'TIMES': r'\*',              # Multiplication operator
    'DIVIDE': r'/',              # Division operator
    'LPAREN': r'\(',             # Left parenthesis
    'RPAREN': r'\)',             # Right parenthesis
    'SKIP': r'[ \t]',            # Skip spaces and tabs
    'MISMATCH': r'.',            # Any other character
}

# Compile a unified regex pattern for the tokenizer
TOKEN_PATTERN = '|'.join(f"(?P<{key}>{pattern})" for key, pattern in TOKEN_TYPES.items())


def tokenize(code):
    """
    Tokenize the input string based on defined token types.

    Args:
        code (str): The input string to tokenize.

    Returns:
        list[tuple[str, Union[int, float, str]]]: A list of (token_type, value) tuples.
    """
    tokens = []
    for match in re.finditer(TOKEN_PATTERN, code):
        kind = match.lastgroup
        value = match.group()
        if kind == 'NUMBER':
            value = float(value) if '.' in value else int(value)
        elif kind == 'SKIP':
            continue
        elif kind == 'MISMATCH':
            raise ValueError(f"Unexpected character: {value!r}")
        tokens.append((kind, value))
    return tokens


class Parser:
    def __init__(self, tokens):
        """
        Initialize the parser with a list of tokens.

        Args:
            tokens (list[tuple[str, Union[int, float, str]]]): Tokens generated from the tokenizer.
        """
        self.tokens = tokens
        self.current_token = None
        self.next_token()

    def next_token(self):
        """Advance to the next token."""
        self.current_token = self.tokens.pop(0) if self.tokens else None

    def parse(self):
        """
        Parse the token stream and evaluate the expression.

        Returns:
            Union[int, float]: The result of the parsed expression.
        """
        return self.expr()

    def eat(self, token_type):
        """
        Consume a token of the expected type or raise an error.

        Args:
            token_type (str): The expected token type.

        Raises:
            ValueError: If the current token type doesn't match the expected type.
        """
        if self.current_token and self.current_token[0] == token_type:
            self.next_token()
        else:
            raise ValueError(
                f"Expected token type {token_type}, but got {self.current_token}"
            )

    def factor(self):
        """
        Parse a factor, which can be a number, a parenthesized expression, or a negated factor.

        Returns:
            Union[int, float]: The value of the factor.
        """
        token = self.current_token
        if token[0] == 'NUMBER':
            self.eat('NUMBER')
            return token[1]
        elif token[0] == 'LPAREN':
            self.eat('LPAREN')
            result = self.expr()
            self.eat('RPAREN')
            return result
        elif token[0] == 'MINUS':
            self.eat('MINUS')
            return -self.factor()
        raise ValueError(f"Unexpected token: {token}")

    def term(self):
        """
        Parse a term, which is a factor optionally followed by multiplication or division.

        Returns:
            Union[int, float]: The value of the term.
        """
        result = self.factor()
        while self.current_token and self.current_token[0] in ('TIMES', 'DIVIDE'):
            token = self.current_token
            if token[0] == 'TIMES':
                self.eat('TIMES')
                result *= self.factor()
            elif token[0] == 'DIVIDE':
                self.eat('DIVIDE')
                result /= self.factor()
        return result

    def expr(self):
        """
        Parse an expression, which is a term optionally followed by addition or subtraction.

        Returns:
            Union[int, float]: The value of the expression.
        """
        result = self.term()
        while self.current_token and self.current_token[0] in ('PLUS', 'MINUS'):
            token = self.current_token
            if token[0] == 'PLUS':
                self.eat('PLUS')
                result += self.term()
            elif token[0] == 'MINUS':
                self.eat('MINUS')
                result -= self.term()
        return result


def get_result(expression):
    """
    Tokenize and parse an expression to evaluate its result.

    Args:
        expression (str): The input mathematical expression.

    Returns:
        Union[int, float]: The evaluated result.
    """
    tokens = tokenize(expression)
    parser = Parser(tokens)
    return parser.parse()


# Usage example (uncomment to test):
# expression = "(400 + (80 / 2) + 450 - (50 / 2)) / 2"
# result = get_result(expression)
# print(f"Result: {result}")  # Should output: 400.0
