use std::{fmt, hash::Hash};

#[derive(Clone)]
pub struct BitSet<'a, T> {
    mapping: &'a [T],
    storage: Box<[u8]>,
}

impl<'a, T> fmt::Debug for BitSet<'a, T>
where
    T: fmt::Debug + PartialEq,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{ ")?;

        for val in self.mapping.iter().filter(|val| self.contains(val)) {
            write!(f, "{val:?} ")?;
        }

        write!(f, "}}")?;

        Ok(())
    }
}

impl<'a, T> PartialEq for BitSet<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.mapping, other.mapping) && self.storage.eq(&other.storage)
    }
}

impl<'a, T> Eq for BitSet<'a, T> {}

impl<'a, T> Hash for BitSet<'a, T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize(self.mapping.as_ptr() as usize);
        state.write(&self.storage);
    }
}

impl<'a, T> PartialOrd for BitSet<'a, T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if !std::ptr::eq(self.mapping, other.mapping) {
            return (self.mapping.as_ptr() as usize)
                .partial_cmp(&(other.mapping.as_ptr() as usize));
        }

        self.storage.partial_cmp(&other.storage)
    }
}

impl<'a, T> Ord for BitSet<'a, T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if !std::ptr::eq(self.mapping, other.mapping) {
            return (self.mapping.as_ptr() as usize).cmp(&(other.mapping.as_ptr() as usize));
        }

        self.storage.cmp(&other.storage)
    }
}

impl<'a, T> BitSet<'a, T> {
    pub fn empty(mapping: &'a [T]) -> BitSet<'a, T> {
        BitSet {
            mapping,
            storage: vec![0; mapping.len() / 8 + if mapping.len() % 8 != 0 { 1 } else { 0 }]
                .into_boxed_slice(),
        }
    }

    pub fn all(mapping: &'a [T]) -> BitSet<'a, T> {
        BitSet {
            mapping,
            storage: vec![255; mapping.len() / 8 + if mapping.len() % 8 != 0 { 1 } else { 0 }]
                .into_boxed_slice(),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T>
    where
        T: PartialEq,
    {
        self.mapping.iter().filter(|v| self.contains(v))
    }

    pub fn mapping(&self) -> &'a [T] {
        self.mapping
    }

    pub fn len(&self) -> usize
    where
        T: PartialEq,
    {
        self.mapping.iter().map(|x| self.contains(x) as usize).sum()
    }

    pub fn remove(&mut self, value: &T) -> bool
    where
        T: PartialEq,
    {
        let Some(index) = self.bit_index_of(value) else { return false };

        if !self.get_bit(index) {
            return false
        }

        self.set_bit(index, false);

        true
    }

    #[allow(unused)]
    pub fn insert(&mut self, value: &T)
    where
        T: PartialEq,
    {
        let index = self.bit_index_of(value).unwrap();

        self.set_bit(index, true);
    }

    pub fn contains(&self, value: &T) -> bool
    where
        T: PartialEq,
    {
        let Some(index) = self.bit_index_of(value) else { return false };

        self.get_bit(index)
    }

    fn set_bit(&mut self, index: usize, value: bool) {
        let mask = 1 << (index % 8);
        if value {
            self.storage[index / 8] |= self.storage[index / 8];
        } else {
            self.storage[index / 8] &= !mask;
        }
    }

    fn get_bit(&self, index: usize) -> bool {
        self.storage[index / 8] & (1 << (index % 8)) != 0
    }

    fn bit_index_of(&self, value: &T) -> Option<usize>
    where
        T: PartialEq,
    {
        self.mapping
            .iter()
            .enumerate()
            .find(|(_, v)| *v == value)
            .map(|(i, _)| i)
    }
}
