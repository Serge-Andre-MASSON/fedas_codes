from hypothesis import given, settings
import hypothesis.strategies as st

from etl.transform.tokenizer import Tokenizer

inner_lists_size = 12
list_list_of_str = st.lists(
    st.lists(
        st.text(
            max_size=8
        ),
        min_size=inner_lists_size,
        max_size=inner_lists_size
    ),
    min_size=1,
    max_size=10
)


@given(list_list_of_str)
@settings(max_examples=5)
def test_tokenizer(list_list_of_str):
    tokenizer = Tokenizer()
    tokenized = tokenizer.fit_transform(list_list_of_str)
    assert len(tokenized) == len(list_list_of_str)
    for seq in tokenized:
        assert len(seq) == inner_lists_size


@given(list_list_of_str)
@settings(max_examples=5)
def test_encoder_decoder(list_list_of_str):
    tokenizer = Tokenizer()
    tokenized = tokenizer.fit_transform(list_list_of_str)
    back_to_str = tokenizer.inverse_transform(tokenized)
    assert list_list_of_str == back_to_str
